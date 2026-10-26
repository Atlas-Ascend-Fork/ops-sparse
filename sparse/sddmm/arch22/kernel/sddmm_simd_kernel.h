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
#include "sddmm_tiling.h"
#include <type_traits>

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
constexpr uint32_t kHugeKCap = 8192;
// Hard ceiling on the resident span. The runtime budget below is usually what
// binds; this only stops a degenerate column range from eating all of UB.
constexpr uint32_t kGroupSpanMax = 512;

// The group path holds R whole A rows in UB at once, so it is only offered for
// reduction dimensions that leave room for them alongside the span.
constexpr uint32_t kGroupACap = 512;
// Per-AIV UB on dav-2201. The span buffer is sized from whatever these
// allocations leave, so this has to be the real number.
constexpr uint32_t kUbBytes = 192u * 1024u;
// Headroom for TPipe's own bookkeeping and for the rounding below.
constexpr uint32_t kUbReserveBytes = 8u * 1024u;
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
constexpr uint32_t kFp32StrideChunk = 256;
constexpr uint32_t kNarrowStrideChunk = 128;
constexpr float kFp16MaxValue = 65504.0f;

__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
{
    if (a == 0) {
        return x;
    }
    return (x + a - 1) / a * a;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

// The `uint64_t mask` argument of the repeat form of Mul is the *连续模式* mask:
// "前面连续的多少个元素参与计算", a count -- the bit-mask form is the separate
// `uint64_t mask[]` overload, and the plain count overload is the 4-argument one.
// Counts above one repeat's worth are clamped to the maximum for the dtype, and a
// negative value is explicitly undefined ("若设成了负值可能产生未定义行为"), which
// is what `~0ull` is as a signed 64-bit value -- so clamp here, never rely on the
// hardware to do it. A short repeat keeps only its first `len` elements live; the
// padding past K is garbage and must not contribute.
__aicore__ inline uint64_t MaskCount(uint32_t len)
{
    return static_cast<uint64_t>((len >= kElemPerRep) ? kElemPerRep : len);
}
} // namespace

// SDDMM over CSR for one (X/Y, C) storage pair of the support matrix.
//
// Compute always accumulates in FP32. Narrow inputs widen at the GM->UB
// boundary, and grouped rows reuse the converted column span. For
// the (float, float) pair each branch below folds away at compile time, leaving
// exactly the code the pre-template fp32 kernel generated.
template <typename XType, typename CType>
class SddmmKernel {
public:
    __aicore__ inline SddmmKernel() {}

    static constexpr bool kXIsF32 = AscendC::IsSameType<XType, float>::value;
    static constexpr bool kCIsF32 = AscendC::IsSameType<CType, float>::value;
    static constexpr uint32_t kStrideChunk = kXIsF32 ? kFp32StrideChunk : kNarrowStrideChunk;
    // FP16 operands feed mixed-precision MulAddDst directly; only BF16 widens
    // its resident span. The saved FP16 space permits a larger row group.
    static constexpr bool kGroupNative = kXIsF32 || AscendC::IsSameType<XType, half>::value;
    using GroupType = typename std::conditional<kGroupNative, XType, float>::type;
    static constexpr uint32_t kGroupRows = AscendC::IsSameType<XType, half>::value ? 16 : 8;
    static constexpr uint32_t kGroupSpanTarget = AscendC::IsSameType<XType, half>::value ? 319 : 183;

    __aicore__ inline void Init(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets, GM_ADDR csrColIndices,
                                GM_ADDR csrValues, const SddmmKernelTilingData& td, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        CopyTiling(td);
        hasGroupHints = td.rowHints != 0;
        if (hasGroupHints) {
            groupHintsGm.SetGlobalBuffer((__gm__ int32_t*)td.rowHints);
        }

        InitGlobalBuffers(matA, matB, csrRowOffsets, csrColIndices, csrValues);

        // Scalar/reduction scratch only needs one K chunk.  The previous fixed
        // kBlockLen allocation reserved 4088 fp32 elements even for the common
        // K=128 anchors, hiding ~32KB of usable UB from the resident path. Keep
        // the full chunk only for genuinely huge-K launches; the batched tile
        // below has its own independent capacity floor.
        const uint32_t innerK = (opAValue == 0) ? matACols : matARows;
        const uint32_t kbuf = AlignUp((innerK < kBlockLen) ? innerK : kBlockLen, kAlign);
        InitReductionBuffers(kbuf);

        // Resolve the layout and the batched tile now, once the buffers that do
        // not depend on them exist. SetupLayout reaches InitStrideOffsets, which
        // writes strideOffsetBuf, so it has to come after that allocation; the
        // staged queues below have to come after this, because they only ever hold
        // one segment and a segment is jTile entries, not kOutStageCap. The old
        // order -- size everything by kOutStageCap, decide the batched plan later
        // inside Process -- is what put 16KB into each of inQueueCsr / outQueueY /
        // colIndBuf for a path that stages at most jTile=64 entries, i.e. exactly
        // the headroom the resident span needs. Process() repeats both calls.
        SetupLayout();
        SetupBatchPlan();

        const uint32_t segCap = InitStagingBuffers();

        const uint32_t accElems = InitProductBuffers(kbuf);

        InitStrideBuffers();

        InitResidentBuffers(kbuf, segCap, accElems);
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets, GM_ADDR csrColIndices,
                                             GM_ADDR csrValues)
    {
        // Leading dimensions describe the physical storage, so a column-major
        // matrix occupies `logical_cols * ld`, while a row-major matrix occupies
        // `logical_rows * ld`. This distinction matters for the padded Yt view:
        // its ld is pitch, which can be larger than K.
        const uint64_t aGmSize = (orderXCol != 0) ? static_cast<uint64_t>(matACols) * ldx :
                                                    static_cast<uint64_t>(matARows) * ldx;
        matAGm.SetGlobalBuffer((__gm__ XType*)matA, aGmSize);
        const uint64_t bGmSize = (orderYCol != 0) ? static_cast<uint64_t>(matBCols) * ldy :
                                                    static_cast<uint64_t>(matBRows) * ldy;
        matBGm.SetGlobalBuffer((__gm__ XType*)matB, bGmSize);
        rowOffsetsGm.SetGlobalBuffer((__gm__ int32_t*)csrRowOffsets, static_cast<uint64_t>(rowCount + 1));
        colIndicesGm.SetGlobalBuffer((__gm__ int32_t*)csrColIndices, nnz);
        csrValuesGm.SetGlobalBuffer((__gm__ CType*)csrValues, nnz);
    }

    __aicore__ inline void InitReductionBuffers(uint32_t kbuf)
    {
        // Doubles as the double-buffered scalar B column pair and as the batched B
        // tile. Rounding the capacity up to a whole repeat is what lets a
        // 64-column run (the longest a dense CSR row usually has) fit one tile:
        // 2 * 4088 = 8176 leaves jTile one column short at K == 128.
        bColFloats = AlignUp(2u * kbuf, kElemPerRep);
        const uint32_t batchFloor = kJTileCap * kElemPerRep;
        if (bColFloats < batchFloor) {
            bColFloats = batchFloor;
        }

        pipe->InitBuffer(aRowBuf, kbuf * sizeof(float));
        pipe->InitBuffer(bColBuf, bColFloats * sizeof(float));
        // `hugeKPath` drives aFullBuf, and both of its uses sit behind an
        // `K > kBlockLen` guard, so it must mean "the reduction dimension does not
        // fit one block" -- not "some operand is long". Testing matARows here made
        // every tall-X shape look huge-K: P-01 is m=8192, K=128, so it never takes
        // the full-cache branch yet still paid for a kHugeKCap-sized aFullBuf, and
        // on the narrow (fp16/bf16) instantiations that pushed the core's UB
        // allocations to ~201KB against a 192KB budget. Resolve K exactly the way
        // SetupLayout does and key the flag off that.
        const uint32_t kInner = (opAValue == 0) ? matACols : matARows;
        hugeKPath = (kInner > kBlockLen);
        if (hugeKPath) {
            pipe->InitBuffer(aFullBuf, AlignUp(kHugeKCap, kAlign) * sizeof(float));
        }
        if constexpr (!kXIsF32) {
            // One scratch buffer serves every narrow X/Y load above. Size it for
            // the widest single load on either path: the huge-K A row, or the
            // doubled B column tile used by the batched column staging.
            const uint32_t hugeElems = AlignUp(kHugeKCap, kAlign);
            const uint32_t batchElems = bColFloats;
            const uint32_t rawElems = hugeElems > batchElems ? hugeElems : batchElems;
            pipe->InitBuffer(xRawBuf, rawElems * sizeof(XType));
        }
    }

    __aicore__ inline uint32_t InitStagingBuffers()
    {
        // Entries one staged buffer must hold. A batched run never exceeds one
        // group's worth of nonzeros; only the scalar path uses the full stage.
        const uint32_t segCap = batchEnabled ? (jTile * kGroupRows) : kOutStageCap;
        // colIndBuf holds one group's staged column indices on the grouped path,
        // so its element capacity is what the group scan may stage.
        groupIndCap = segCap;

        pipe->InitBuffer(inQueueCsr, 1, segCap * sizeof(float));
        // FP16 saturation compares whole vector repeats, including a short
        // segment's inactive tail. Reserve that tail inside this queue slot.
        const uint32_t outCap = AscendC::IsSameType<CType, half>::value ? AlignUp(segCap, kElemPerRep) : segCap;
        pipe->InitBuffer(outQueueY, 1, outCap * sizeof(float));
        if constexpr (!kCIsF32) {
            // C-boundary staging for the narrow rows of the support matrix. The
            // fp32 pair neither needs nor allocates it.
            pipe->InitBuffer(cRawBuf, segCap * sizeof(CType));
            // These three buffers are only reached by the scalar/non-batched
            // path. Keeping them out of the batched layout gives the resident
            // path room for a second raw slot without changing its interface.
            if (!batchEnabled) {
                pipe->InitBuffer(outQueueC, 1, kAlign * sizeof(CType));
                pipe->InitBuffer(cScalarBuf, kAlign * sizeof(CType));
                pipe->InitBuffer(cScalarWideBuf, kAlign * sizeof(float));
            }
        }
        pipe->InitBuffer(colIndBuf, segCap * sizeof(int32_t));
        // Group formation needs only the row boundaries for its small window.
        // Pull them with one DMA instead of issuing two scalar GM reads per row.
        if (batchEnabled) {
            pipe->InitBuffer(groupRowBuf, AlignUp(kGroupRows + 1u, kAlign) * sizeof(int32_t));
            if (hasGroupHints) {
                pipe->InitBuffer(groupHintBuf, AlignUp(kGroupRows, kAlign) * 4);
            }
        }
        return segCap;
    }

    __aicore__ inline uint32_t InitProductBuffers(uint32_t kbuf)
    {
        // mulBuf holds either one K-length product vector (scalar path) or
        // kJTileCap product rows of kElemPerRep each (batched path). Huge-K
        // execution can never enter the batched path, so reserving the full
        // tile there would push the narrow instantiations beyond the 192KB UB
        // limit once aFullBuf is present.
        // The product tile is sized for the *selected* jTile, not the hard
        // ceiling.  At K=128 bColFloats admits 64 columns, so reserving the
        // 128-column maximum wastes 16KB of UB on every core and needlessly
        // shrinks the resident-column window below.
        const uint32_t batchMulElems = batchEnabled ? (jTile * kElemPerRep) : kbuf;
        mulElems = hugeKPath ? kbuf : ((batchMulElems > kbuf) ? batchMulElems : kbuf);
        pipe->InitBuffer(mulBuf, mulElems * sizeof(float));
        // DotProduct is reachable only on the non-batched path. Do not reserve
        // its scalar reduction scratch when the resident/batched path is active;
        // those 544 bytes are part of the double-buffer budget below.
        if (!batchEnabled) {
            pipe->InitBuffer(reduceTmpBuf, kbuf * sizeof(float));
            pipe->InitBuffer(reduceDstBuf, kAlign * sizeof(float));
        }
        // Dot-product accumulator, kept separate from reduceTmpBuf because
        // ReduceSum overwrites its sharedTmpBuffer argument.
        const uint32_t batchScratchElems = batchEnabled ? AlignUp(jTile, kAlign) : AlignUp(kJTileCap, kAlign);
        const uint32_t accElems = batchEnabled ? batchScratchElems * (kAlign + 1) : batchScratchElems;
        pipe->InitBuffer(accBuf, accElems * sizeof(float));
        return accElems;
    }

    __aicore__ inline void InitStrideBuffers()
    {
        // The strided gather is only reachable when a row of X or a column of Y is
        // non-contiguous in GM; skip its staging buffers otherwise so contiguous
        // shapes (notably the huge-K path) keep the UB headroom.
        const bool aCtg = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        const bool bCtg = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        needStrideGather = (!aCtg || !bCtg);
        if (needStrideGather) {
            pipe->InitBuffer(stridePadBuf, kStrideChunk * kStrideElemBytes);
            pipe->InitBuffer(strideOffsetBuf, kStrideChunk * sizeof(int32_t));
        }
    }

    __aicore__ inline uint32_t InitGroupFixedBuffers(uint32_t kbuf, uint32_t segCap, uint32_t accElems)
    {
        uint32_t fixed = kbuf * sizeof(float); // aRowBuf
        fixed += bColFloats * sizeof(float);   // bColBuf
        if (hugeKPath) {
            fixed += AlignUp(kHugeKCap, kAlign) * sizeof(float); // aFullBuf
        }
        if constexpr (!kXIsF32) {
            const uint32_t hugeElems = AlignUp(kHugeKCap, kAlign);
            const uint32_t rawElems = hugeElems > bColFloats ? hugeElems : bColFloats;
            fixed += rawElems * sizeof(XType); // xRawBuf
        }
        const uint32_t outCap = AscendC::IsSameType<CType, half>::value ? AlignUp(segCap, kElemPerRep) : segCap;
        fixed += (segCap + outCap) * sizeof(float); // inQueueCsr, outQueueY
        if constexpr (!kCIsF32) {
            fixed += segCap * sizeof(CType);
            fixed += 2u * kAlign * sizeof(CType) + kAlign * sizeof(float);
        }
        fixed += segCap * sizeof(int32_t);                           // colIndBuf
        fixed += AlignUp(kGroupRows + 1u, kAlign) * sizeof(int32_t); // groupRowBuf
        if (hasGroupHints) {
            fixed += AlignUp(kGroupRows, kAlign) * 4;
        }
        fixed += mulElems * sizeof(float);
        if (!batchEnabled) {
            fixed += kbuf * sizeof(float) + 2u * kAlign * sizeof(float);
        }
        fixed += accElems * sizeof(float);
        if (needStrideGather) {
            fixed += kStrideChunk * kStrideElemBytes + kStrideChunk * sizeof(int32_t);
        }
        // One whole A row per group member, so R*K floats.
        const uint32_t aBytes = kGroupRows * K * sizeof(GroupType);
        pipe->InitBuffer(aGroupBuf, aBytes);
        fixed += aBytes;

        return fixed;
    }

    __aicore__ inline uint32_t InitGroupDoubleBuffer(uint32_t free, uint32_t perCol)
    {
        // A narrow resident span is converted through raw scratch. Prefer a
        // two-slot pipeline: the next DMA can fill one slot while V casts the
        // other. If the extra slot would remove the eight-row window, retain
        // the single-slot implementation rather than silently changing the
        // group's row reuse factor.
        if constexpr (!kGroupNative) {
            const uint32_t hugeElems = AlignUp(kHugeKCap, kAlign);
            const uint32_t rawElems = hugeElems > bColFloats ? hugeElems : bColFloats;
            const uint32_t rawBytes = rawElems * static_cast<uint32_t>(sizeof(XType));
            const uint32_t freeDouble = (free > rawBytes) ? (free - rawBytes) : 0u;
            const uint64_t doubleBufferBytes = static_cast<uint64_t>(perCol) * kGroupSpanTarget;
            if (perCol != 0 && static_cast<uint64_t>(freeDouble) >= doubleBufferBytes) {
                pipe->InitBuffer(xRawBuf2, rawBytes);
                free = freeDouble;
                groupRawDouble = true;
                // Allocate one event per direction and per raw slot. The
                // slot-local events are what allow MTE2 to fill slot N+1
                // while V casts slot N; a single helper event would wait for
                // the newest DMA and serialize the two pipes again.
                for (uint32_t i = 0; i < 2; ++i) {
                    groupRawMte2V[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
                    groupRawVMte2[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
                }
            }
        }
        return free;
    }

    __aicore__ inline void InitResidentBuffers(uint32_t kbuf, uint32_t segCap, uint32_t accElems)
    {
        // Resident-column-span path. Sized last because it gets the remainder:
        // every allocation above is a deterministic function of the tiling, so
        // summing them beats guessing a constant, and guessing wrong here is a
        // hard UB-overflow abort rather than a slow kernel.
        // The group loads R whole A rows as one contiguous transfer, which needs
        // X row-major, untransposed, and tightly packed -- a padded ld would put
        // gaps between the rows and the single DMA would splice the padding in.
        // Every VEC operand the group feeds -- aGrp[i*K] to Mul, yStage[outOff]
        // to Muls -- is a UB address, and dav-2201 requires those to be 32B
        // aligned. Rows land at i*K floats in the group's A block and at the
        // running nonzero count in the output block, so both strides have to be
        // whole blocks or the vector unit faults on an unaligned base. Gating on
        // it here is what keeps that a shape property rather than a runtime
        // surprise; a padded A stride would lift the K condition, but no
        // performance anchor needs it (all K == 128).
        if (batchEnabled && aContig && ldx == K && (K * sizeof(GroupType)) % kDmaAlignBytes == 0 && K <= kGroupACap) {
            uint32_t fixed = InitGroupFixedBuffers(kbuf, segCap, accElems);
            // Budget the resident operand dtype. BF16 reuses xRawBuf in bounded
            // chunks while widening; FP16 keeps its original storage precision.
            const uint32_t perCol = kPitch * static_cast<uint32_t>(sizeof(GroupType));
            uint32_t free = (kUbBytes > kUbReserveBytes + fixed) ? (kUbBytes - kUbReserveBytes - fixed) : 0u;
            free = InitGroupDoubleBuffer(free, perCol);
            uint32_t cap = free / perCol;
            if (cap >= jTile) {
                if (cap > kGroupSpanMax) {
                    cap = kGroupSpanMax;
                }
                if (cap > kGroupSpanTarget) {
                    cap = kGroupSpanTarget;
                }
                spanCap = cap;
                groupEnabled = true;
                pipe->InitBuffer(bSpanBuf, spanCap * kPitch * sizeof(GroupType));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (SetupLayout()) {
            SetupBatchPlan();
            if (batchEnabled) {
                RunBatchedNnzPartition();
            } else if (nnzPartition != 0) {
                RunNnzPartition();
            } else {
                RunRowPartition();
            }
        }
        if (groupRawDouble) {
            for (uint32_t i = 0; i < 2; ++i) {
                pipe->ReleaseEventID<AscendC::HardEvent::MTE2_V>(groupRawMte2V[i]);
                pipe->ReleaseEventID<AscendC::HardEvent::V_MTE2>(groupRawVMte2[i]);
            }
        }
    }

private:
    // Only partition boundaries need a GM binary search. Interior rows retain
    // the grouped DMA path; long rows are split into independent vector tiles.
    __aicore__ inline uint32_t RowOfNnz(uint32_t offset)
    {
        uint32_t lo = 0;
        uint32_t hi = rowCount;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            const uint32_t end = static_cast<uint32_t>(rowOffsetsGm.GetValue(mid + 1) - idxBaseValue);
            if (end <= offset) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    __aicore__ inline void RunBatchedNnzPartition()
    {
        // Each core owns whole output DMA blocks, except for the final tail.
        // This also prevents two narrow stores from sharing one 32-byte block.
        constexpr uint32_t outAlign = kDmaAlignBytes / sizeof(CType);
        const uint32_t units = (static_cast<uint32_t>(nnz) + outAlign - 1) / outAlign;
        const uint32_t blockCount = AscendC::GetBlockNum();
        const uint32_t cores = blockCount == 0 ? 1u : blockCount;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t per = units / cores;
        const uint32_t rem = units % cores;
        uint32_t first = (core * per + MinU32(core, rem)) * outAlign;
        uint32_t last = MinU32(first + (per + (core < rem ? 1u : 0u)) * outAlign, static_cast<uint32_t>(nnz));
        uint32_t row, finalRow;
        if (hasGroupHints) {
            auto bounds = groupHintBuf.Get<int32_t>();
            AscendC::DataCopy(bounds, groupHintsGm[AlignUp(rowCount, 8u) + core * 8], 8);
            SyncMte2ToS();
            row = static_cast<uint32_t>(bounds.GetValue(0));
            finalRow = static_cast<uint32_t>(bounds.GetValue(1));
            first = static_cast<uint32_t>(bounds.GetValue(2));
            last = static_cast<uint32_t>(bounds.GetValue(3));
        } else {
            if (first >= last) {
                return;
            }
            row = RowOfNnz(first);
            finalRow = RowOfNnz(last - 1);
        }
        if (first >= last) {
            return;
        }
        const uint32_t rowEnd = static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseValue);
        if (row == finalRow) {
            ProcessRowRange(row, first, last);
            return;
        }
        const uint32_t rowStart = static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseValue);
        if (first != rowStart) {
            ProcessRowRange(row, first, rowEnd);
            ++row;
        }
        const uint32_t finalEnd = static_cast<uint32_t>(rowOffsetsGm.GetValue(finalRow + 1) - idxBaseValue);
        if (last == finalEnd) {
            RunRowPartitionGrouped(row, finalRow + 1);
        } else {
            RunRowPartitionGrouped(row, finalRow);
            const uint32_t finalStart = static_cast<uint32_t>(rowOffsetsGm.GetValue(finalRow) - idxBaseValue);
            ProcessRowRange(finalRow, finalStart, last);
        }
    }

    // Split the reduction dimension K and the contiguity flags out of the tiling
    // data. Returns false when there is nothing to do or the shapes disagree.
    __aicore__ inline bool SetupLayout()
    {
        if (nnz == 0 || rowCount == 0) {
            return false;
        }
        const uint32_t innerA = (opAValue == 0) ? matACols : matARows;
        const uint32_t innerB = (opBValue == 0) ? matBRows : matBCols;
        if (innerA != innerB) {
            return false;
        }
        K = innerA;
        colCount = (opBValue == 0) ? matBCols : matBRows;
        // A row (resp. B column) is contiguous in GM when the memory order and the
        // requested transpose agree.
        aContig = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        bContig = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        // The strided gather (and hence its offset table) is only reachable when a
        // row of X or a column of Y is non-contiguous in GM.
        if (needStrideGather) {
            InitStrideOffsets();
        }
        return SetupDmaFlags();
    }

    // Each core takes a contiguous slice of nonzeros; used for huge-K shapes where
    // a single row would not fill the cores.
    __aicore__ inline void RunNnzPartition()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t extraN = (coreIdx < remainderNnz) ? 1u : 0u;
        const uint32_t myNnz = nnzPerCore + extraN;
        const uint32_t myBase = coreIdx * nnzPerCore + ((coreIdx < remainderNnz) ? coreIdx : remainderNnz);
        for (uint32_t j = 0; j < myNnz; ++j) {
            if (myBase + j < nnz) {
                ProcessNnz(myBase + j);
            }
        }
    }

    // Each core takes a contiguous slice of rows (the common case).
    __aicore__ inline void RunRowPartition()
    {
        const uint32_t totalRows = rowCount;
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t extra = (coreIdx < remainderRows) ? 1u : 0u;
        const uint32_t myCount = rowsPerCore + extra;
        const uint32_t myBase = coreIdx * rowsPerCore + ((coreIdx < remainderRows) ? coreIdx : remainderRows);
        const uint32_t myEnd = MinU32(myBase + myCount, totalRows);
        RunRowPartitionGrouped(myBase, myEnd);
    }

    // Contiguous small-K loads are only worth a DMA above a size threshold.
    __aicore__ inline bool SetupDmaFlags()
    {
        const uint32_t totalRows = rowCount;
        const bool kInRange = (K <= kBlockLen) && (K >= kSmallDmaMinK);
        smallKContigADma = aContig && kInRange && (totalRows >= kSmallDmaMinRows);
        smallKContigBDma = bContig && kInRange;
        return true;
    }
    // Copy the scalar tiling fields into members.
    __aicore__ inline void CopyTiling(const SddmmKernelTilingData& td)
    {
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
        idxBaseValue = static_cast<int32_t>(td.idxBase);
    }
    // Decide whether the batched (vector-reduction) path is usable, and size its tile.
    //
    // The batched path stages jTile B columns contiguously in UB as a jTile x kPitch
    // matrix, multiplies them against the cached A row using a repeat-stride-0
    // broadcast, then collapses each staged row to a scalar with a single
    // WholeReduceSum. That removes the per-nonzero ReduceSum + GetValue vector-to-scalar
    // stall that dominates the scalar path.
    __aicore__ inline void SetupBatchPlan()
    {
        batchEnabled = false;
        // The batched path relies on a full 32B block for each staged row. For
        // very small reductions that padding dominates the useful payload and,
        // on dav-2201, the narrow-type Cast of sub-block tiles is not reliable.
        // Keep those shapes on the scalar path; performance anchors all use
        // K >= 64 and retain the bulk path.
        if (!bContig || K < kSmallDmaMinK || K > kBlockLen) {
            return;
        }
        // A staged B column is the whole ldy-long row of a column-major B, and a
        // run of consecutive columns is DMA'd straight into back-to-back UB
        // slots (dstStride == 0). So the tile pitch has to be exactly ldy, and
        // ldy has to be a whole number of 32B blocks for the vector repeat
        // strides below to address it.
        if (ldy < K || (ldy % kAlign) != 0) {
            return;
        }
        kPitch = ldy;
        // src1RepStride / dstRepStride are 8-bit block counts.
        if ((kPitch / kAlign) > kMaxPitchBlocks) {
            return;
        }
        const uint32_t cap = MinU32(bColFloats / kPitch, kJTileCap);
        if (cap == 0) {
            return;
        }
        jTile = cap;
        batchEnabled = true;
        // A column of B is contiguous here, so one multi-block DMA per run of
        // consecutive columns beats one DMA per column whenever the run is longer
        // than one. The scan is a handful of scalar reads, so take it always
        // rather than guessing at the density.
        runBatchable = true;
    }

    // Build the byte-offset table 0, 32, 64, ... used by the strided gather below.
    // Computed once per kernel invocation.
    __aicore__ inline void InitStrideOffsets()
    {
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::CreateVecIndex<int32_t>(offs, 0, kStrideChunk);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kStrideElemBytes), static_cast<int32_t>(kStrideChunk));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Strided GM->UB load: DataCopyPad pulls `len` single-element blocks (each
    // padded to 32B by the blockCount form), then one Gather compacts them into
    // ub[0..len). Vectorised throughout — no per-element scalar access.
    __aicore__ inline void GatherChunk(AscendC::LocalTensor<float>& ub, AscendC::GlobalTensor<XType>& gm, uint64_t off,
                                       uint32_t stride, uint32_t len)
    {
        if (stride == 1u) {
            CopyContig(ub, gm, off, len);
            return;
        }
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
        // Every element lands in its own 32B block, so the byte-offset table is
        // shared between the fp32 and narrow paths.
        const uint32_t srcGapBytes = (stride - 1u) * (uint32_t)sizeof(XType);
        if constexpr (kXIsF32) {
            AscendC::LocalTensor<float> pad32 = stridePadBuf.Get<float>();
            for (uint32_t base = 0; base < len; base += kStrideChunk) {
                const uint32_t cnt = MinU32(len - base, kStrideChunk);
                AscendC::DataCopyExtParams cp{static_cast<uint16_t>(cnt), (uint32_t)sizeof(float), srcGapBytes, 0, 0};
                AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], cp, pd);
                SyncMte2ToV();
                AscendC::Gather<float>(ub[base], pad32, offsU, 0, cnt);
                AscendC::PipeBarrier<PIPE_V>();
                // pad32 is reused by the next chunk's DMA (and by the next call),
                // so MTE2 has to wait for the gather that reads it.
                SyncVToMte2();
            }
        } else {
            AscendC::LocalTensor<XType> pad32 = stridePadBuf.Get<XType>();
            AscendC::LocalTensor<XType> compact = xRawBuf.Get<XType>();
            for (uint32_t base = 0; base < len; base += kStrideChunk) {
                const uint32_t cnt = MinU32(len - base, kStrideChunk);
                AscendC::DataCopyExtParams cp{static_cast<uint16_t>(cnt), (uint32_t)sizeof(XType), srcGapBytes, 0, 0};
                AscendC::DataCopyPadExtParams<XType> pd{false, 0, 0, static_cast<XType>(0)};
                AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], cp, pd);
                SyncMte2ToV();
                AscendC::Gather<XType>(compact[base], pad32, offsU, 0, cnt);
                AscendC::PipeBarrier<PIPE_V>();
                // Same WAR on pad32 as above.
                SyncVToMte2();
            }
            AscendC::Cast<float, XType>(ub, compact, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // GM->UB contiguous load, widening XType to fp32. For the fp32 pair this
    // copies straight into `ub`, which is the pre-template behaviour exactly.
    //
    // Contract for every GM->UB loader here (this one, CopyBRun, GatherChunk):
    // the destination is *reclaimed*, not freshly handed over, so both
    // directions of the dependency need an event.
    //
    //   entry: the vector pipe may still be reading what this buffer held --
    //          the A row the last ReduceBTile broadcast from, the B tile it
    //          just reduced, the previous iteration's staging -- so fence
    //          V->MTE2 before the overwriting DMA;
    //   exit : the data must be visible to the vector unit, so fence MTE2->V.
    //
    // Both fences are unconditional and live here rather than at the call sites
    // because the call sites cannot know whether their destination is fresh:
    // aRowBuf / bColBuf are reused by LoadARow, StageBTile, DotProduct and
    // ProcessNnz, and xRawBuf is shared scratch behind all of them. The older
    // shape of this function had only the exit fence on the narrow branch (and
    // none at all on the fp32 branch), which "worked" purely by accident: the
    // narrow branch's trailing V->MTE2 happened to fence the *next* call's DMA
    // too, so fp16 was safe and fp32 was not. Two separate bugs came out of
    // that -- the dot-product loop reading a stale bC (no MTE2->V) and the
    // batched path overwriting the still-in-use A row (no V->MTE2).
    __aicore__ inline void CopyContig(AscendC::LocalTensor<float>& ub, AscendC::GlobalTensor<XType>& gm, uint64_t off,
                                      uint32_t len)
    {
        SyncVToMte2();
        if constexpr (kXIsF32) {
            const uint32_t lenBytes = len * (uint32_t)sizeof(float);
            if ((lenBytes & kDmaAlignMask) == 0 && lenBytes > 0) {
                AscendC::DataCopy(ub, gm[off], len);
            } else {
                AscendC::DataCopyExtParams p{1, lenBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(ub, gm[off], p, pad);
            }
            SyncMte2ToV();
        } else {
            // Narrow elements are never 32B-aligned in a useful way, so always
            // take the padded path and widen in UB. The entry fence above also
            // covers xRawBuf, whose previous occupant is the Cast of the call
            // before this one.
            AscendC::LocalTensor<XType> raw = xRawBuf.Get<XType>();
            AscendC::DataCopyExtParams p{1, len * (uint32_t)sizeof(XType), 0, 0, 0};
            AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
            AscendC::DataCopyPad(raw, gm[off], p, pad);
            SyncMte2ToV();
            AscendC::Cast<float, XType>(ub, raw, AscendC::RoundMode::CAST_NONE, len);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Stage `run` consecutive B columns into `run` adjacent UB slots of kPitch
    // elements each, as a single multi-block DMA.
    //
    // With B column-major, column c is the ldy-long span [c * ldy, c * ldy + ldy),
    // so consecutive columns are back-to-back and one multi-block DMA covers the
    // whole run: srcStride == 0, and dstStride == 0 lands the blocks on the same
    // ldy pitch in UB. Reading the padded tail of the row (k >= K) is harmless --
    // the reduction never looks past K.
    __aicore__ inline void CopyBRun(AscendC::LocalTensor<float> dst, uint64_t gmOff, uint32_t run)
    {
        if (run == 1) {
            CopyContig(dst, matBGm, gmOff, K);
            return;
        }
        // Same reclaim contract as CopyContig: fence the destination (and the
        // shared xRawBuf) against the vector pipe before writing, then make the
        // result visible to it.
        SyncVToMte2();
        if constexpr (kXIsF32) {
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), kPitch * (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(dst, matBGm[gmOff], p, pad);
            SyncMte2ToV();
        } else {
            AscendC::LocalTensor<XType> raw = xRawBuf.Get<XType>();
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), kPitch * (uint32_t)sizeof(XType), 0, 0, 0};
            AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
            AscendC::DataCopyPad(raw, matBGm[gmOff], p, pad);
            SyncMte2ToV();
            AscendC::Cast<float, XType>(dst, raw, AscendC::RoundMode::CAST_NONE, run * kPitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
    }

    __aicore__ inline void SyncVToMte2()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev);
    }

    __aicore__ inline void SyncVToS()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ev);
    }

    __aicore__ inline void SyncSToV()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ev);
    }

    // ---- C boundary ---------------------------------------------------------
    // CType appears nowhere above this line. Every read widens to fp32 and every
    // write narrows back, so the whole algorithm stays fp32-typed and the
    // (float, float) instantiation is the pre-template code unchanged.

    // bisheng has no scalar bf16<->fp32 conversion ("not support bf16 type
    // cast"), so both directions of the narrow scalar path stage a single
    // element through UB and use the vector Cast. Same idiom as spmv/arch22.
    __aicore__ inline float LoadCsrScalar(uint64_t off)
    {
        if constexpr (kCIsF32) {
            return csrValuesGm.GetValue(off);
        } else {
            AscendC::LocalTensor<CType> raw = cScalarBuf.Get<CType>();
            AscendC::LocalTensor<float> wide = cScalarWideBuf.Get<float>();
            raw.SetValue(0, csrValuesGm.GetValue(off));
            SyncSToV();
            AscendC::Cast<float, CType>(wide, raw, AscendC::RoundMode::CAST_NONE, 1);
            SyncVToS();
            return wide.GetValue(0);
        }
    }

    __aicore__ inline void StoreCsrScalar(uint64_t off, float value)
    {
        if constexpr (kCIsF32) {
            AscendC::LocalTensor<float> y = outQueueY.AllocTensor<float>();
            y.SetValue(0, value);
            outQueueY.EnQue(y);
            AscendC::LocalTensor<float> yo = outQueueY.DeQue<float>();
            AscendC::DataCopyExtParams op{1, (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(csrValuesGm[off], yo, op);
            outQueueY.FreeTensor(yo);
        } else {
            // One narrow element: same queue-staged DMA, just a narrower tensor.
            AscendC::LocalTensor<float> wide = cScalarWideBuf.Get<float>();
            AscendC::LocalTensor<CType> y = outQueueC.AllocTensor<CType>();
            if constexpr (AscendC::IsSameType<CType, half>::value) {
                value = value > kFp16MaxValue ? kFp16MaxValue : (value < -kFp16MaxValue ? -kFp16MaxValue : value);
            }
            wide.SetValue(0, value);
            SyncSToV();
            AscendC::Cast<CType, float>(y, wide, AscendC::RoundMode::CAST_RINT, 1);
            outQueueC.EnQue(y);
            AscendC::LocalTensor<CType> yo = outQueueC.DeQue<CType>();
            AscendC::DataCopyExtParams op{1, (uint32_t)sizeof(CType), 0, 0, 0};
            AscendC::DataCopyPad(csrValuesGm[off], yo, op);
            outQueueC.FreeTensor(yo);
        }
    }

    __aicore__ inline void LoadCsrChunk(AscendC::LocalTensor<float>& dst, uint64_t off, uint32_t n)
    {
        AscendC::DataCopyExtParams p{1, n * (uint32_t)sizeof(CType), 0, 0, 0};
        if constexpr (kCIsF32) {
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(dst, csrValuesGm[off], p, pad);
        } else {
            AscendC::LocalTensor<CType> raw = cRawBuf.Get<CType>();
            // The previous chunk's MTE3 may still be reading this shared raw
            // buffer. Do not let the next MTE2 load overwrite it early.
            SyncMte3ToMte2();
            AscendC::DataCopyPadExtParams<CType> pad{false, 0, 0, static_cast<CType>(0)};
            AscendC::DataCopyPad(raw, csrValuesGm[off], p, pad);
            SyncMte2ToV();
            AscendC::Cast<float, CType>(dst, raw, AscendC::RoundMode::CAST_NONE, n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Reuse the finished B tile for clamp scratch. Compare/Select keeps NaNs
    // unchanged while finite overflow and infinities saturate like the scalar path.
    __aicore__ inline void SaturateFp16Chunk(AscendC::LocalTensor<float>& values, uint32_t n)
    {
        if constexpr (AscendC::IsSameType<CType, half>::value) {
            const uint32_t count = AlignUp(n, kElemPerRep);
            AscendC::LocalTensor<float> limits = bColBuf.Get<float>();
            AscendC::LocalTensor<uint8_t> mask = limits[kOutStageCap].ReinterpretCast<uint8_t>();
            AscendC::Duplicate(limits, kFp16MaxValue, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(mask, values, limits, AscendC::CMPMODE::GT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(values, mask, limits, values, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(limits, -kFp16MaxValue, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(mask, values, limits, AscendC::CMPMODE::LT, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(values, mask, limits, values, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreCsrChunk(uint64_t off, AscendC::LocalTensor<float>& src, uint32_t n)
    {
        AscendC::DataCopyExtParams p{1, n * (uint32_t)sizeof(CType), 0, 0, 0};
        if constexpr (kCIsF32) {
            AscendC::DataCopyPad(csrValuesGm[off], src, p);
        } else {
            AscendC::LocalTensor<CType> raw = cRawBuf.Get<CType>();
            // Consecutive beta-zero chunks have no intervening MTE2 load, so
            // explicitly protect the prior store before vector code reuses raw.
            SyncMte3ToV();
            SaturateFp16Chunk(src, n);
            AscendC::Cast<CType, float>(raw, src, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
            SyncVToMte3();
            AscendC::DataCopyPad(csrValuesGm[off], raw, p);
        }
    }

    __aicore__ inline float DotProduct(uint64_t aOff, uint32_t aStride, uint64_t bOff, uint32_t bStride, int aCache)
    {
        AscendC::LocalTensor<float> aC = aRowBuf.Get<float>();
        AscendC::LocalTensor<float> bC = bColBuf.Get<float>();
        AscendC::LocalTensor<float> ml = mulBuf.Get<float>();
        AscendC::LocalTensor<float> rt = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> rd = reduceDstBuf.Get<float>();
        float acc = 0.0f;
        const bool bUseDma = (bStride == 1u && (aCache == kACacheFull || smallKContigBDma));
        for (uint32_t base = 0; base < K; base += kBlockLen) {
            const uint32_t len = (K - base < kBlockLen) ? (K - base) : kBlockLen;
            AscendC::LocalTensor<float> aOp = aC;
            if (aCache == kACacheNone) {
                GatherChunk(aC, matAGm, aOff + static_cast<uint64_t>(base) * aStride, aStride, len);
            } else if (aCache == kACacheFull) {
                aOp = aFullBuf.Get<float>()[base];
            }
            if (bUseDma) {
                // bC is the same UB slot on every iteration; CopyContig fences
                // both directions of that reuse itself (a PipeBarrier<PIPE_MTE2>
                // used to stand here and ordered MTE2 only against MTE2, which
                // provided neither the WAR nor the RAW).
                CopyContig(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, len);
            } else {
                GatherChunk(bC, matBGm, bOff + static_cast<uint64_t>(base) * bStride, bStride, len);
            }
            AscendC::Mul(ml, aOp, bC, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float, true>(rd, ml, rt, static_cast<int32_t>(len));
            SyncVToS();
            acc += rd.GetValue(0);
        }
        return acc;
    }

    __aicore__ inline void ProcessNnz(uint32_t gnnz)
    {
        uint32_t lo = 0;
        uint32_t hi = rowCount;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (rowOffsetsGm.GetValue(mid + 1) - idxBaseValue <= static_cast<int32_t>(gnnz)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const uint32_t row = lo;
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
        const int32_t col = colIndicesGm.GetValue(gnnz) - idxBaseValue;
        float dot = 0.0f;
        if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
            const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy : static_cast<uint64_t>(col);
            const uint32_t bStr = bContig ? 1u : ldy;
            dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
        }
        // Same beta guard as the segment paths: reading C when it only gets
        // multiplied by zero is wasted traffic, and a caller that hands us a
        // write-only buffer would get NaN out of `0 * garbage`.
        const float cVal = (betaValue != 0.0f) ? LoadCsrScalar(gnnz) : 0.0f;
        StoreCsrScalar(gnnz, alphaValue * dot + betaValue * cVal);
    }

    // Load the A row (or A column when opA/order make it strided) into UB once per row.
    __aicore__ inline int LoadARow(uint64_t aOff, uint32_t aStride)
    {
        if (alphaValue == 0.0f) {
            return kACacheNone;
        }
        if (K <= kBlockLen) {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            if (smallKContigADma || (batchEnabled && aStride == 1u)) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                GatherChunk(a, matAGm, aOff, aStride, K);
            }
            return kACacheSingle;
        }
        if (hugeKPath && K <= kHugeKCap) {
            AscendC::LocalTensor<float> a = aFullBuf.Get<float>();
            if (aStride == 1u) {
                CopyContig(a, matAGm, aOff, K);
            } else {
                GatherChunk(a, matAGm, aOff, aStride, K);
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
    __aicore__ inline void ComputeSegmentBatched(AscendC::LocalTensor<int32_t>& colInd, uint32_t segBase,
                                                 uint32_t segNnz, uint64_t bBB, AscendC::LocalTensor<float>& aOp,
                                                 AscendC::LocalTensor<float>& acc)
    {
        AscendC::LocalTensor<float> bTile = bColBuf.Get<float>();
        StageBTile(colInd, segBase, segNnz, bBB, bTile);
        ReduceBTile(bTile, aOp, acc, segNnz);
    }

    // Stage this segment's B columns into `bTile` as segNnz rows of kPitch each.
    // CSR column indices ascend, so runs of consecutive columns are common; each
    // run is issued as one multi-block DMA instead of one DMA per nonzero.
    // Out-of-range columns contribute zero.
    __aicore__ inline void StageBTile(AscendC::LocalTensor<int32_t>& colInd, uint32_t segBase, uint32_t segNnz,
                                      uint64_t bBB, AscendC::LocalTensor<float>& bTile)
    {
        // Every destination in this segment is disjoint, and bColBuf/xRawBuf are
        // reclaimed only once the previous segment's reduction has finished.
        // Fence that reuse once for the whole tile instead of once per nonzero.
        SyncVToMte2();
        AscendC::LocalTensor<XType> rawTile;
        if constexpr (!kXIsF32) {
            rawTile = xRawBuf.Get<XType>();
        }
        uint32_t j = 0;
        while (j < segNnz) {
            const int32_t col = colInd.GetValue(segBase + j) - idxBaseValue;
            if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                if constexpr (kXIsF32) {
                    AscendC::Duplicate(bTile[j * kPitch], 0.0f, static_cast<int32_t>(kPitch));
                } else {
                    AscendC::Duplicate(rawTile[j * kPitch], static_cast<XType>(0), static_cast<int32_t>(kPitch));
                }
                ++j;
                continue;
            }
            uint32_t run = 1;
            if (runBatchable) {
                while (j + run < segNnz &&
                       colInd.GetValue(segBase + j + run) - idxBaseValue == col + static_cast<int32_t>(run)) {
                    ++run;
                }
            }
            const uint64_t gmOff = bBB + static_cast<uint64_t>(col) * ldy;
            if constexpr (kXIsF32) {
                CopyBRunNoFence(bTile[j * kPitch], gmOff, run);
            } else {
                CopyBRunNoFence(rawTile[j * kPitch], gmOff, run);
            }
            j += run;
        }
        // Make all staged data visible before the one vector Cast (narrow path)
        // or the reduction (fp32 path). The narrow raw tile is intentionally
        // converted in one instruction so xRawBuf is no longer a per-column
        // synchronization point.
        SyncMte2ToV();
        if constexpr (!kXIsF32) {
            AscendC::Cast<float, XType>(bTile, rawTile, AscendC::RoundMode::CAST_NONE, segNnz * kPitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // A run of consecutive columns is one *linear* span on both sides: column c
    // is the ldy-long span [c * ldy, c * ldy + ldy), so consecutive columns are
    // back to back in GM, and the staged rows are back to back in UB. Describing
    // that as {blockCount = run, srcStride = 0, dstStride = 0} states "this is one
    // contiguous range" in the most expensive way available: the DMA engine still
    // walks `run` separate blocks. asc-doc is explicit on the split --
    // "对于连续搬运场景，推荐使用连续数据搬运(DataCopy)" -- so collapse the run
    // into a single descriptor. The per-block cost is what made fp16 measure *no
    // faster* than fp32 on these anchors (630us vs 615us): identical block count,
    // half the bytes, so the payload never mattered.
    //
    // kPitch is simultaneously the GM stride between staged columns, the UB
    // stride between staged rows, and (times the element size) the length of every
    // block in the run, so a whole number of 32B blocks is exactly the condition
    // that puts both offsets and the length on the DMA alignment.
    __aicore__ inline bool ContiguousRunDma() const
    {
        return batchEnabled && kPitch != 0 && ((kPitch * static_cast<uint32_t>(sizeof(XType))) & kDmaAlignMask) == 0;
    }

    // Batch staging primitive. The caller owns one V->MTE2 and one MTE2->V
    // fence around the complete segment, so this helper must only enqueue the
    // actual DMA. For a single column the same padded copy handles K tails;
    // consecutive columns use one contiguous transfer.
    template <typename DstType>
    __aicore__ inline void CopyBRunNoFence(AscendC::LocalTensor<DstType> dst, uint64_t gmOff, uint32_t run)
    {
        if (run > 1 && ContiguousRunDma()) {
            AscendC::DataCopy(dst, matBGm[gmOff], run * kPitch);
            return;
        }
        if constexpr (kXIsF32) {
            if (run == 1) {
                const uint32_t lenBytes = K * static_cast<uint32_t>(sizeof(float));
                AscendC::DataCopyExtParams p{1, lenBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(dst, matBGm[gmOff], p, pad);
            } else {
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), kPitch * static_cast<uint32_t>(sizeof(float)),
                                             0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(dst, matBGm[gmOff], p, pad);
            }
        } else {
            AscendC::LocalTensor<XType> raw = dst;
            if (run == 1) {
                const uint32_t lenBytes = K * static_cast<uint32_t>(sizeof(XType));
                AscendC::DataCopyExtParams p{1, lenBytes, 0, 0, 0};
                AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
                AscendC::DataCopyPad(raw, matBGm[gmOff], p, pad);
            } else {
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(run), kPitch * static_cast<uint32_t>(sizeof(XType)),
                                             0, 0, 0};
                AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
                AscendC::DataCopyPad(raw, matBGm[gmOff], p, pad);
            }
        }
    }

    // Multiply the staged tile by the cached A row and collapse each staged row to
    // one dot product. acc[0..segNnz) receives the results.
    template <typename OperandType>
    __aicore__ inline void ReduceBTile(AscendC::LocalTensor<OperandType>& bTile, AscendC::LocalTensor<OperandType>& aOp,
                                       AscendC::LocalTensor<float>& acc, uint32_t segNnz)
    {
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
        mp.src1RepStride = static_cast<uint8_t>(kPitch * sizeof(OperandType) / kDmaAlignBytes);

        // Keep one 64-lane FP32 accumulator per output, independent of K.
        // Half inputs use the documented half*half+float instruction directly.
        uint32_t first = 0;
        if constexpr (AscendC::IsSameType<OperandType, float>::value) {
            AscendC::Mul(prod, aOp, bTile, MaskCount(K), rep, mp);
            first = kElemPerRep;
        } else {
            AscendC::Duplicate(prod, 0.0f, static_cast<int32_t>(segNnz * kElemPerRep));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t base = first; base < K; base += kElemPerRep) {
            const uint32_t len = MinU32(K - base, kElemPerRep);
            AscendC::MulAddDst(prod, aOp[base], bTile[base], MaskCount(len), rep, mp);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (K >= kElemPerRep && segNnz % kAlign == 0) {
            // [outputs,64] -> [outputs,8] -> [outputs]: each second-stage
            // repeat finishes eight outputs without serial whole-row reductions.
            auto partial = accBuf.Get<float>()[AlignUp(jTile, kAlign)];
            AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(partial, prod, kElemPerRep, segNnz, 1, 1, repStride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceDataBlock<AscendC::ReduceType::SUM>(acc, partial, kElemPerRep, segNnz / kAlign, 1, 1,
                                                               repStride);
        } else {
            AscendC::WholeReduceSum<float>(acc, prod, static_cast<int32_t>(MinU32(K, kElemPerRep)),
                                           static_cast<int32_t>(segNnz), 1, 1, repStride);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Scalar fallback for layouts the batched path cannot serve (strided B, huge K).
    //
    // The dot products go straight into the caller's output stage: this path serves
    // whole segments, i.e. up to kOutStageCap entries, while accBuf is only sized
    // for the batched tile (kJTileCap). Accumulating here instead of in accBuf is
    // what keeps a wide segment from running off the end of that buffer.
    __aicore__ inline void ComputeSegmentScalar(AscendC::LocalTensor<int32_t>& colInd, uint32_t segBase,
                                                uint32_t segNnz, uint64_t bBB, uint64_t aOff, uint32_t aStride,
                                                int aCache, AscendC::LocalTensor<float>& csrStage,
                                                AscendC::LocalTensor<float>& out)
    {
        for (uint32_t j = 0; j < segNnz; ++j) {
            const int32_t col = colInd.GetValue(segBase + j) - idxBaseValue;
            float dot = 0.0f;
            if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
                const uint64_t bOff = bContig ? (bBB + static_cast<uint64_t>(col) * ldy) :
                                                (bBB + static_cast<uint64_t>(col));
                const uint32_t bStr = bContig ? 1u : ldy;
                dot = DotProduct(aOff, aStride, bOff, bStr, aCache);
            }
            const float cv = betaValue != 0.0f ? csrStage.GetValue(j) : 0.0f;
            out.SetValue(j, alphaValue * dot + betaValue * cv);
        }
    }

    // alpha == 0 case: y = beta * C over the row, in segments of kOutStageCap.
    __aicore__ inline void ScaleRowOnly(uint64_t rowOff, uint32_t rowNnz)
    {
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += kOutStageCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, kOutStageCap);
            const uint64_t segOff = rowOff + segBase;
            AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
            if (betaValue == 0.0f) {
                AscendC::Duplicate(yStage, 0.0f, static_cast<int32_t>(AlignUp(segNnz, kAlign)));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
                LoadCsrChunk(csrIn, segOff, segNnz);
                inQueueCsr.EnQue(csrIn);
                AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();
                AscendC::Muls(yStage, csrStage, betaValue, static_cast<int32_t>(segNnz));
                AscendC::PipeBarrier<PIPE_V>();
                inQueueCsr.FreeTensor(csrStage);
            }
            outQueueY.EnQue(yStage);
            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            StoreCsrChunk(segOff, yOut, segNnz);
            outQueueY.FreeTensor(yOut);
        }
    }

    // ---- resident column span ------------------------------------------------
    //
    // Every row of a performance anchor is 64 *consecutive* columns and the window
    // slides by 17 columns from one row to the next, so staging each row's B tile
    // separately re-reads the same columns ~3.8 times. That redundancy, not the
    // arithmetic, is what the kernel is made of: fp16 moves half the bytes of fp32
    // on these shapes and measures *slower* (630us vs 615us), i.e. the payload is
    // irrelevant next to the per-DMA and per-row fence cost. At the anchors' share
    // of bandwidth the staged path wants 134MB / ~167us for P-01 against a kernel
    // budget near 88us, so no amount of tuning the existing staging reaches the
    // target -- the reuse has to come out of the algorithm.
    //
    // So process R consecutive rows as one unit and keep the *union* of their
    // columns resident in UB. Under the delivered pattern R rows span
    // 17*(R-1) + 64 columns, so R=8 keeps 183 columns resident instead of staging
    // 8*64 = 512 -- 0.36x the B traffic -- and pays one fence pair per group in
    // place of ~15 per row. Each row then reduces straight out of the span: its
    // columns are consecutive, so its window is a contiguous slice of the span and
    // ReduceBTile's existing kPitch repeat stride addresses it unchanged.
    //
    // Consecutive columns are a property of the delivered pattern rather than of
    // CSR at large, so every row is checked and the group stops at the first row
    // that fails. Rejected rows go down the original per-row path.
    __aicore__ inline uint32_t ProcessGroupFrom(uint32_t g, uint32_t hi)
    {
        if (!groupEnabled || alphaValue == 0.0f) {
            return 0;
        }
        // Form the group before touching the A/B load. gStart[i] is row i's first
        // nonzero, gStart[R] closes the last row, and gC0[i] is its first column.
        //
        // The scan reads its columns out of UB, not GM. Reading them from GM is
        // what made this path latency-bound rather than throughput-bound: two
        // endpoint columns per row on top of the two row offsets is four scalar
        // round trips to HBM per row, and at K=64..128 a group moves so little
        // data (P-01: 46.8KB) that the scalar latency, not the transfer, sets the
        // pace. One 2KB column-index DMA replaces 16 GM round trips per group and
        // every subsequent read is a UB access.
        uint32_t gStart[kGroupRows + 1];
        int32_t gC0[kGroupRows];
        int32_t minC = 0;
        int32_t maxC = 0;
        uint32_t R = 0;
        uint32_t limit = MinU32(kGroupRows, hi - g);
        AscendC::LocalTensor<int32_t> hints;
        auto colInd = colIndBuf.Get<int32_t>();
        limit = StageGroupIndices(g, limit, gStart, hints, colInd);
        R = SelectGroupRows(limit, gStart, gC0, minC, maxC, hints, colInd);
        if (R == 0) {
            return 0;
        }
        const uint32_t span = static_cast<uint32_t>(maxC - minC + 1);
        const uint32_t outBase = gStart[0];
        const uint32_t totalNnz = gStart[R] - outBase;

        if (!ValidateGroupColumns(gStart, gC0, R, outBase, totalNnz)) {
            return 0;
        }

        LoadGroupOperands(g, R, minC, span);

        StoreGroupResults(gStart, gC0, R, minC, outBase, totalNnz);
        return R;
    }

    __aicore__ inline void PipelineGroupSpan(AscendC::LocalTensor<GroupType> bSpan, uint32_t spanElems, int32_t minC,
                                             uint32_t& pendingOff, uint32_t& pendingCnt, uint32_t& pendingSlot)
    {
        auto raw0 = xRawBuf.Get<XType>();
        auto raw1 = xRawBuf2.Get<XType>();
        const uint32_t rawCap = bColFloats;
        AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
        uint32_t chunk = 0;
        for (uint32_t off = 0; off < spanElems; off += rawCap, ++chunk) {
            const uint32_t cnt = MinU32(spanElems - off, rawCap);
            const uint32_t slot = chunk & 1u;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(groupRawVMte2[slot]);
            AscendC::LocalTensor<XType> raw = (slot == 0u) ? raw0 : raw1;
            AscendC::DataCopyExtParams p{1, cnt * static_cast<uint32_t>(sizeof(XType)), 0, 0, 0};
            AscendC::DataCopyPad(raw, matBGm[static_cast<uint64_t>(minC) * ldy + off], p, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(groupRawMte2V[slot]);
            if (pendingCnt != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(groupRawMte2V[pendingSlot]);
                AscendC::LocalTensor<XType> pendingRaw = (pendingSlot == 0u) ? raw0 : raw1;
                AscendC::Cast<float, XType>(bSpan[pendingOff], pendingRaw, AscendC::RoundMode::CAST_NONE,
                                            static_cast<int32_t>(pendingCnt));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(groupRawVMte2[pendingSlot]);
            }
            pendingOff = off;
            pendingCnt = cnt;
            pendingSlot = slot;
        }
    }

    __aicore__ inline uint32_t StageGroupIndices(uint32_t g, uint32_t limit, uint32_t* gStart,
                                                 AscendC::LocalTensor<int32_t>& hints,
                                                 AscendC::LocalTensor<int32_t> colInd)
    {
        AscendC::LocalTensor<int32_t> rowBuf = groupRowBuf.Get<int32_t>();
        AscendC::DataCopyExtParams rowCp{1, (limit + 1u) * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> rowPad{false, 0, 0, 0};
        AscendC::DataCopyPad(rowBuf, rowOffsetsGm[g], rowCp, rowPad);
        if (hasGroupHints) {
            hints = groupHintBuf.Get<int32_t>();
            AscendC::DataCopyPad(hints, groupHintsGm[g], {1, limit * 4, 0, 0, 0}, rowPad);
        }
        SyncMte2ToS();
        gStart[0] = static_cast<uint32_t>(rowBuf.GetValue(0) - idxBaseValue);
        // Stage only the rows whose index block fits the buffer; a row that
        // overflows it ends the group rather than the DMA.
        uint32_t stage = 0;
        for (uint32_t i = 0; i < limit; ++i) {
            const int32_t end = rowBuf.GetValue(i + 1) - idxBaseValue;
            const uint32_t n = static_cast<uint32_t>(end) - gStart[i];
            // Rows are written into the group's output block back to back at the
            // running nonzero count, so a row whose count is not a whole number of
            // 32B blocks would put the next row's Muls on an unaligned UB base.
            // The delivered pattern's rows are all 64, i.e. always whole blocks.
            if (n == 0 || n > jTile || (n % kAlign) != 0 || n > groupIndCap - stage) {
                limit = i;
                break;
            }
            gStart[i + 1] = static_cast<uint32_t>(end);
            stage += n;
        }
        if (limit == 0) {
            return 0;
        }
        if (!hasGroupHints) {
            AscendC::DataCopyExtParams cp{1, stage * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
            AscendC::DataCopyPad(colInd, colIndicesGm[gStart[0]], cp, cpd);
        }
        if (!hasGroupHints) {
            SyncMte2ToS();
        }
        return limit;
    }

    __aicore__ inline uint32_t SelectGroupRows(uint32_t limit, const uint32_t* gStart, int32_t* gC0, int32_t& minC,
                                               int32_t& maxC, AscendC::LocalTensor<int32_t> hints,
                                               AscendC::LocalTensor<int32_t> colInd)
    {
        uint32_t R = 0;
        uint32_t rel = 0;
        for (uint32_t i = 0; i < limit; ++i) {
            const uint32_t n = gStart[i + 1] - gStart[i];
            const int32_t c0 = hasGroupHints ? hints.GetValue(i) : colInd.GetValue(rel) - idxBaseValue;
            const int32_t c1 = hasGroupHints ? c0 + static_cast<int32_t>(n) - 1 :
                                               colInd.GetValue(rel + n - 1) - idxBaseValue;
            rel += n;
            if (c0 < 0 || static_cast<uint32_t>(c1) >= colCount) {
                break;
            }
            if (c1 - c0 + 1 != static_cast<int32_t>(n)) {
                break;
            }
            const int32_t lo = (R == 0) ? c0 : (c0 < minC ? c0 : minC);
            const int32_t up = (R == 0) ? c1 : (c1 > maxC ? c1 : maxC);
            if (static_cast<uint32_t>(up - lo + 1) > spanCap) {
                break;
            }
            minC = lo;
            maxC = up;
            gC0[R] = c0;
            ++R;
        }
        return R;
    }

    __aicore__ inline bool ValidateGroupColumns(const uint32_t* gStart, const int32_t* gC0, uint32_t R,
                                                uint32_t outBase, uint32_t totalNnz)
    {
        auto colInd = colIndBuf.Get<int32_t>();
        // Descriptor-owned hints prove contiguity of the validated snapshot.
        // Otherwise verify every index; matching endpoints alone are insufficient.
        if (!hasGroupHints) {
            auto sequence = bColBuf.Get<int32_t>();
            auto expected = mulBuf.Get<int32_t>();
            AscendC::CreateVecIndex(sequence, int32_t(0), totalNnz);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < R; ++i) {
                const uint32_t off = gStart[i] - outBase;
                AscendC::Adds(expected[off], sequence[off], gC0[i] + idxBaseValue - static_cast<int32_t>(off),
                              gStart[i + 1] - gStart[i]);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(expected, colInd, expected, totalNnz);
            AscendC::PipeBarrier<PIPE_V>();
            auto errors = bColBuf.Get<float>();
            AscendC::Cast(errors, expected, AscendC::RoundMode::CAST_NONE, totalNnz);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(errors, errors, totalNnz);
            AscendC::PipeBarrier<PIPE_V>();
            auto checked = accBuf.Get<float>();
            AscendC::ReduceMax(checked, errors, mulBuf.Get<float>(), totalNnz, false);
            SyncVToS();
            if (checked.GetValue(0) != 0.0f) {
                return false;
            }
        }

        return true;
    }

    __aicore__ inline void LoadNarrowGroupSpan(AscendC::LocalTensor<GroupType> bSpan, uint32_t spanElems, int32_t minC)
    {
        auto raw = xRawBuf.Get<XType>();
        const uint32_t rawCap = bColFloats;
        AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
        if (groupRawDouble) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(groupRawVMte2[slot]);
            }
            // Two raw slots turn the per-chunk dependency chain into a
            // pipeline: issue DMA for chunk N+1 before waiting for and
            // casting chunk N. A slot is reclaimed only after its previous
            // Cast, protected by V->MTE2 on the next lap.
            AscendC::LocalTensor<XType> raw0 = xRawBuf.Get<XType>();
            AscendC::LocalTensor<XType> raw1 = xRawBuf2.Get<XType>();
            uint32_t pendingOff = 0;
            uint32_t pendingCnt = 0;
            uint32_t pendingSlot = 0;
            uint32_t chunk = 0;
            PipelineGroupSpan(bSpan, spanElems, minC, pendingOff, pendingCnt, pendingSlot);
            if (pendingCnt != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(groupRawMte2V[pendingSlot]);
                AscendC::LocalTensor<XType> pendingRaw = (pendingSlot == 0u) ? raw0 : raw1;
                AscendC::Cast<float, XType>(bSpan[pendingOff], pendingRaw, AscendC::RoundMode::CAST_NONE,
                                            static_cast<int32_t>(pendingCnt));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(groupRawVMte2[pendingSlot]);
            }
            // Drain both slots before raw0 is reused for A or by a row
            // fallback. No ready token survives a group boundary.
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(groupRawVMte2[slot]);
            }
        } else {
            for (uint32_t off = 0; off < spanElems; off += rawCap) {
                const uint32_t cnt = MinU32(spanElems - off, rawCap);
                // xRawBuf is reclaimed after each conversion. The first
                // iteration is covered by the group-entry fence; later ones
                // need the matching WAR fence after the preceding Cast.
                if (off != 0) {
                    SyncVToMte2();
                }
                AscendC::DataCopyExtParams p{1, cnt * static_cast<uint32_t>(sizeof(XType)), 0, 0, 0};
                AscendC::DataCopyPad(raw, matBGm[static_cast<uint64_t>(minC) * ldy + off], p, pad);
                SyncMte2ToV();
                AscendC::Cast<float, XType>(bSpan[off], raw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void LoadGroupOperands(uint32_t g, uint32_t R, int32_t minC, uint32_t span)
    {
        // One fence pair covers the whole group: the B span, the A rows and (when
        // beta != 0) the C block. This is the change that matters -- the staged
        // path pays a pair per row.
        SyncVToMte2();
        // FP32 transfers overlap before one MTE2->V fence. Narrow inputs
        // reuse xRawBuf in bounded chunks while converting the span, then reuse
        // that same raw buffer for the group's A rows; this trades one tiny A DMA
        // serialization for enough UB to keep several overlapping rows resident.
        AscendC::LocalTensor<GroupType> bSpan = bSpanBuf.Get<GroupType>();
        const uint32_t spanElems = span * kPitch;
        if constexpr (kGroupNative) {
            AscendC::DataCopyExtParams p{1, spanElems * static_cast<uint32_t>(sizeof(GroupType)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<GroupType> pad{false, 0, 0, static_cast<GroupType>(0)};
            // Columns are consecutive, so the span is one linear run in GM.
            AscendC::DataCopyPad(bSpan, matBGm[static_cast<uint64_t>(minC) * ldy], p, pad);
        }
        AscendC::LocalTensor<GroupType> aGrp = aGroupBuf.Get<GroupType>();
        const uint32_t aElems = R * K;
        if constexpr (kGroupNative) {
            AscendC::DataCopyExtParams p{1, aElems * static_cast<uint32_t>(sizeof(GroupType)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<GroupType> pad{false, 0, 0, static_cast<GroupType>(0)};
            AscendC::DataCopyPad(aGrp, matAGm[static_cast<uint64_t>(g) * ldx], p, pad);
        }
        if constexpr (!kGroupNative) {
            AscendC::LocalTensor<XType> raw = xRawBuf.Get<XType>();
            const uint32_t rawCap = bColFloats;
            AscendC::DataCopyPadExtParams<XType> pad{false, 0, 0, static_cast<XType>(0)};
            LoadNarrowGroupSpan(bSpan, spanElems, minC);
            // Reuse the same narrow buffer for A only after the final B Cast has
            // retired. A is small (R*K), so the extra DMA cannot dominate the
            // saved resident-span capacity.
            SyncVToMte2();
            AscendC::DataCopyExtParams p{1, aElems * static_cast<uint32_t>(sizeof(XType)), 0, 0, 0};
            AscendC::DataCopyPad(raw, matAGm[static_cast<uint64_t>(g) * ldx], p, pad);
            SyncMte2ToV();
            AscendC::Cast<float, XType>(aGrp, raw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(aElems));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreGroupResults(const uint32_t* gStart, const int32_t* gC0, uint32_t R, int32_t minC,
                                             uint32_t outBase, uint32_t totalNnz)
    {
        auto bSpan = bSpanBuf.Get<GroupType>();
        auto aGrp = aGroupBuf.Get<GroupType>();
        AscendC::LocalTensor<float> csrStage;
        const bool hasC = (betaValue != 0.0f);
        if (hasC) {
            csrStage = inQueueCsr.AllocTensor<float>();
            LoadCsrChunk(csrStage, outBase, totalNnz);
            inQueueCsr.EnQue(csrStage);
            csrStage = inQueueCsr.DeQue<float>();
        }
        SyncMte2ToV();

        // Per row: window the span, reduce, scatter into the group's output block.
        // The group's outputs are one contiguous run in C, so a single store
        // closes it out.
        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        for (uint32_t i = 0; i < R; ++i) {
            const uint32_t n = gStart[i + 1] - gStart[i];
            const uint32_t outOff = gStart[i] - outBase;
            AscendC::LocalTensor<GroupType> aOp = aGrp[i * K];
            AscendC::LocalTensor<float> yDst = yStage[outOff];
            // The row's window is a contiguous slice of the resident span,
            // and kPitch is a whole number of 32B blocks, so this base is a legal
            // vector operand with no copy at all.
            AscendC::LocalTensor<GroupType> bSrc = bSpan[static_cast<uint32_t>(gC0[i] - minC) * kPitch];
            ReduceBTile(bSrc, aOp, yDst, n);
        }
        // Rows occupy disjoint, aligned slices. Apply the identical FP32
        // alpha/beta operations once to the contiguous group rather than
        // issuing two vector instructions and barriers for every short row.
        if (alphaValue != 1.0f) {
            AscendC::Muls(yStage, yStage, alphaValue, static_cast<int32_t>(totalNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (hasC) {
            AscendC::Axpy<float, float>(yStage, csrStage, betaValue, static_cast<int32_t>(totalNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (hasC) {
            inQueueCsr.FreeTensor(csrStage);
        }
        outQueueY.EnQue(yStage);
        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        StoreCsrChunk(outBase, yOut, totalNnz);
        outQueueY.FreeTensor(yOut);
    }

    // Row-partition loop with the group path in front of the per-row one.
    __aicore__ inline void RunRowPartitionGrouped(uint32_t lo, uint32_t hi)
    {
        uint32_t g = lo;
        while (g < hi) {
            const uint32_t used = ProcessGroupFrom(g, hi);
            if (used == 0) {
                ProcessRow(g);
                ++g;
            } else {
                g += used;
            }
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const int32_t start = rowOffsetsGm.GetValue(row) - idxBaseValue;
        const int32_t end = rowOffsetsGm.GetValue(row + 1) - idxBaseValue;
        ProcessRowRange(row, static_cast<uint32_t>(start), static_cast<uint32_t>(end));
    }

    __aicore__ inline void ProcessRowRange(uint32_t row, uint32_t start, uint32_t end)
    {
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
        const int aCache = LoadARow(aOff, aStride);
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += segCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, segCap);
            ProcessSegment(static_cast<uint64_t>(start) + segBase, segNnz, 0ULL, aOff, aStride, aCache);
        }
    }

    // Load one segment's column indices (and C values when beta != 0), compute the
    // dot products, then write back y = alpha * dot + beta * C.
    __aicore__ inline void ProcessSegment(uint64_t segOff, uint32_t segNnz, uint64_t bBB, uint64_t aOff,
                                          uint32_t aStride, int aCache)
    {
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        AscendC::DataCopyExtParams cp{1, segNnz * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
        AscendC::DataCopyPad(colInd, colIndicesGm[segOff], cp, cpd);

        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        if (betaValue != 0.0f) {
            LoadCsrChunk(csrIn, segOff, segNnz);
        }
        // colInd is walked by the scalar unit (ComputeSegmentScalar / StageBTile),
        // csrStage by the vector unit, so both DMAs have to land first. The
        // non-batched branch used to run a PipeBarrier<PIPE_MTE2> here instead,
        // which orders MTE2 against MTE2 and left the scalar read of colInd with
        // no MTE2->S event at all; SyncMte2ToS also orders the csr DMA that
        // LoadCsrChunk issued above, so one sync serves both branches.
        SyncMte2ToS();
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();

        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        if (batchEnabled) {
            AscendC::LocalTensor<float> acc = accBuf.Get<float>();
            AscendC::LocalTensor<float> aOp = (aCache == kACacheFull) ? aFullBuf.Get<float>() : aRowBuf.Get<float>();
            ComputeSegmentBatched(colInd, 0, segNnz, bBB, aOp, acc);
            CombineBatched(yStage, acc, csrStage, segNnz);
        } else {
            // accBuf is sized for the batched tile only; the scalar path writes its
            // dot products into yStage directly.
            ComputeSegmentScalar(colInd, 0, segNnz, bBB, aOff, aStride, aCache, csrStage, yStage);
        }
        inQueueCsr.FreeTensor(csrStage);

        outQueueY.EnQue(yStage);
        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        StoreCsrChunk(segOff, yOut, segNnz);
        outQueueY.FreeTensor(yOut);
    }

    // y = alpha * acc + beta * csr, entirely in the vector pipe.
    __aicore__ inline void CombineBatched(AscendC::LocalTensor<float>& yStage, AscendC::LocalTensor<float>& acc,
                                          AscendC::LocalTensor<float>& csrStage, uint32_t segNnz)
    {
        AscendC::Muls(yStage, acc, alphaValue, static_cast<int32_t>(segNnz));
        AscendC::PipeBarrier<PIPE_V>();
        if (betaValue != 0.0f) {
            AscendC::Axpy<float, float>(yStage, csrStage, betaValue, static_cast<int32_t>(segNnz));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe* pipe = nullptr;
    // Resident column span plus the group's A rows. Both only exist when the
    // grouped path is usable; `spanCap` is 0 and `groupEnabled` false otherwise.
    // bSpanBuf holds every dtype as FP32, so the
    // reduction reads it directly and no per-row window copy is needed.
    AscendC::TBuf<AscendC::TPosition::VECCALC> bSpanBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> groupRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> groupHintBuf;
    AscendC::GlobalTensor<int32_t> groupHintsGm;
    bool hasGroupHints = false;
    // Narrow span transfers are widened in bounded chunks through xRawBuf, so
    // the resident path does not reserve a second span-sized raw allocation.
    AscendC::TBuf<AscendC::TPosition::VECCALC> aGroupBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFullBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colIndBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridePadBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideOffsetBuf;
    // Narrow-precision staging for the GM->UB X/Y loads. Only allocated when X/Y
    // are not fp32; the loads are serialised, so one scratch buffer serves them all.
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRawBuf;
    // The resident span opts into this second slot only when the UB budget can
    // preserve the full eight-row window. It lets the next narrow DMA overlap the
    // current span Cast instead of serialising every chunk.
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRawBuf2;
    // Same idea for the C write-back path, plus a matching single-element queue.
    AscendC::TBuf<AscendC::TPosition::VECCALC> cRawBuf;
    // One-element staging for the scalar C conversions, which bisheng can only
    // do on the vector unit.
    AscendC::TBuf<AscendC::TPosition::VECCALC> cScalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cScalarWideBuf;
    bool needStrideGather = false;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueCsr;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueC;
    AscendC::GlobalTensor<XType> matAGm;
    AscendC::GlobalTensor<XType> matBGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::GlobalTensor<int32_t> colIndicesGm;
    AscendC::GlobalTensor<CType> csrValuesGm;
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
    bool runBatchable = false;
    uint32_t nnzPartition = 0;
    uint32_t nnzPerCore = 0;
    uint32_t remainderNnz = 0;
    uint32_t ldx = 0;
    uint32_t ldy = 0;
    uint32_t orderXCol = 0;
    uint32_t orderYCol = 0;
    // Added to every raw rowOffsets / colIndices value before it is used as a
    // 0-based offset or column id, so 0- and 1-based CSR share one code path.
    int32_t idxBaseValue = 0;
    // Element capacity of bColBuf, i.e. one staged B tile's worth of floats.
    uint32_t bColFloats = 0;
    // Element capacity of mulBuf; ReduceBTile checks the per-chunk product slabs
    // against it before choosing the pipeline-friendly layout.
    uint32_t mulElems = 0;
    uint32_t kPitch = 0;
    uint32_t jTile = 0;
    // Resident-column-span path: whether it is usable at all, and how many
    // consecutive columns bSpanBuf can hold.
    bool groupEnabled = false;
    bool groupRawDouble = false;
    event_t groupRawMte2V[2] = {};
    event_t groupRawVMte2[2] = {};
    uint32_t spanCap = 0;
    // Element capacity of colIndBuf, i.e. how many column indices one group may
    // stage for its contiguity scan.
    uint32_t groupIndCap = 0;
};

#endif // SDDMM_ARCH22_SIMD_KERNEL_H_
