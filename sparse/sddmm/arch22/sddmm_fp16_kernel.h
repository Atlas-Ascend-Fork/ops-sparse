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

#ifndef SDDMM_ARCH22_FP16_KERNEL_H_
#define SDDMM_ARCH22_FP16_KERNEL_H_

class KernelSparseSddmmFp16 {
public:
    __aicore__ inline KernelSparseSddmmFp16() {}

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
        matAGm.SetGlobalBuffer((__gm__ half *)matA,
                               static_cast<uint64_t>(matARows) * ldx);
        const uint64_t bGmSize = static_cast<uint64_t>(matBRows) *
                                   ((ldy > matBCols) ? ldy : matBCols);
        matBGm.SetGlobalBuffer((__gm__ half *)matB, bGmSize);
        rowOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)csrRowOffsets,
                                     static_cast<uint64_t>(rowCount + 1));
        colIndicesGm.SetGlobalBuffer((__gm__ int32_t *)csrColIndices, nnz);
        csrValuesGm.SetGlobalBuffer((__gm__ half *)csrValues, nnz);
    }

    __aicore__ inline void InitBuffers() {
        const uint32_t kbuf = AlignUp(kBlockLen, kAlign);
        pipe->InitBuffer(aRowBuf, kbuf * sizeof(float));
        pipe->InitBuffer(bColBuf, kbuf * sizeof(float));
        hugeKPath = (matARows > kBlockLen) || (matACols > kBlockLen);
        pipe->InitBuffer(inQueueCsr, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(outQueueY, 1, kOutStageCap * sizeof(float));
        pipe->InitBuffer(colIndBuf, kOutStageCap * sizeof(int32_t));
        pipe->InitBuffer(mulBuf, AlignUp(kbuf, kElemPerRep) * sizeof(float));
        pipe->InitBuffer(reduceTmpBuf, kbuf * sizeof(float));
        pipe->InitBuffer(reduceDstBuf, kAlign * sizeof(float));
        pipe->InitBuffer(halfBuf, AlignUp(kBlockLen, static_cast<uint32_t>(16)) * sizeof(half));
        // One aligned reduction slot per nonzero; compact a segment only after
        // all dot products have completed, keeping results in the vector pipe.
        pipe->InitBuffer(dotStageBuf, kVectorSegmentCap * kAlign * sizeof(float));
        pipe->InitBuffer(dotOffsetBuf, kVectorSegmentCap * sizeof(int32_t));
        pipe->InitBuffer(clampMaskBuf, kVectorSegmentCap / 8);
        // The strided gather is only reachable when a row of X or a column of Y is
        // non-contiguous in GM; skip its staging buffers otherwise so contiguous
        // shapes (notably the huge-K path) keep the UB headroom.
        const bool aCtg = (orderXCol == 0 && opAValue == 0) || (orderXCol == 1 && opAValue == 1);
        const bool bCtg = (orderYCol == 0 && opBValue == 1) || (orderYCol == 1 && opBValue == 0);
        needStrideGather = (!aCtg || !bCtg);
        if (needStrideGather) {
            pipe->InitBuffer(stridePadBuf, kFp16StrideChunk * kStrideElemBytes);
            pipe->InitBuffer(strideOffsetBuf, kFp16StrideChunk * sizeof(int32_t));
        }
    }


    __aicore__ inline void Process() {
        if (!SetupLayout()) {
            return;
        }
        if (nnzPartition != 0) {
            RunNnzPartition();
        } else {
            RunRowPartition();
        }
    }

private:
    friend struct SddmmRowPartition<KernelSparseSddmmFp16>;
    friend struct SddmmCommonLayout<KernelSparseSddmmFp16>;

    // Split the reduction dimension K and the contiguity flags out of the tiling
    // data. Returns false when there is nothing to do or the shapes disagree.
    __aicore__ inline bool SetupLayout() {
        if (!SddmmCommonLayout<KernelSparseSddmmFp16>::Setup(*this)) {
            return false;
        }
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
    __aicore__ inline void RunNnzPartition() {
        const SddmmWorkRange range = GetSddmmWorkRange(nnzPerCore, remainderNnz);
        for (uint32_t j = 0; j < range.count; ++j) {
            if (range.begin + j < nnz) {
                ProcessNnz(range.begin + j);
            }
        }
    }

    // Each core takes a contiguous slice of rows (the common case).
    __aicore__ inline void RunRowPartition() {
        SddmmRowPartition<KernelSparseSddmmFp16>::Run(*this);
    }

    // The FP16 path has no small-K DMA specialisation; nothing extra to set up.
    __aicore__ inline bool SetupDmaFlags() {
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
    // Byte-offset table for the strided gather. Gather indexes in element units
    // of the gathered type, so for half the 32B block pitch is 16 elements.
    __aicore__ inline void InitStrideOffsets() {
        AscendC::LocalTensor<int32_t> offs = strideOffsetBuf.Get<int32_t>();
        const uint32_t count = MinU32(K, kFp16StrideChunk);
        AscendC::CreateVecIndex<int32_t>(offs, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offs, offs, static_cast<int32_t>(kStrideElemBytes),
                      static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SyncMte2ToV() {
        SyncSddmmEvent<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void SyncVToMte2() {
        SyncSddmmEvent<AscendC::HardEvent::V_MTE2>();
    }

    __aicore__ inline void SyncMte2ToS() {
        SyncSddmmEvent<AscendC::HardEvent::MTE2_S>();
    }

    __aicore__ inline void SyncVToS() {
        SyncSddmmEvent<AscendC::HardEvent::V_S>();
    }

    __aicore__ inline void SyncSToV() {
        SyncSddmmEvent<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void SyncVToMte3() {
        SyncSddmmEvent<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void SyncSToMte3() {
        SyncSddmmEvent<AscendC::HardEvent::S_MTE3>();
    }

    __aicore__ inline void SyncMte3ToS() {
        SyncSddmmEvent<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void SyncMte3ToV() {
        SyncSddmmEvent<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void SyncMte3ToMte2() {
        SyncSddmmEvent<AscendC::HardEvent::MTE3_MTE2>();
    }

    // This private implementation fragment is included in the class because
    // every routine operates on this kernel's UB buffers and state.
#include "sddmm_fp16_data_path.inc"

    static constexpr float FP16_MAX_VALUE = 65504.0f;

    __aicore__ inline float Saturate(float v) {
        if (v > FP16_MAX_VALUE) {
            return FP16_MAX_VALUE;
        }
        if (v < -FP16_MAX_VALUE) {
            return -FP16_MAX_VALUE;
        }
        return v;
    }


    __aicore__ inline void ProcessNnz(uint32_t gnnz) {
        const uint32_t row = FindSddmmRowForNnz(rowOffsetsGm, rowCount, gnnz);
        const uint64_t aOff = aContig ? static_cast<uint64_t>(row) * ldx : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;
        int aCache = kACacheNone;
        if (alphaValue != 0.0f && K <= kBlockLen) {
            aCache = kACacheSingle;
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            Gather16(a, matAGm, aOff, aStride, K);
        }
        const int32_t col = colIndicesGm.GetValue(gnnz);
        float dot = 0.0f;
        if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
            const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy
                                          : static_cast<uint64_t>(col);
            dot = DotProduct(aOff, aStride, bOff, bContig ? 1u : ldy, aCache);
        }
        const float cVal = betaValue != 0.0f ? static_cast<float>(csrValuesGm.GetValue(gnnz)) : 0.0f;
        AscendC::LocalTensor<float> y = outQueueY.AllocTensor<float>();
        AcquireScalarOutput();
        AscendC::LocalTensor<half> yh = y.ReinterpretCast<half>();
        yh.SetValue(0, static_cast<half>(Saturate(alphaValue * dot + betaValue * cVal)));
        StoreScalarHalf(gnnz, 1, yh);
        outQueueY.FreeTensor(y);
    }

    // Stage the row's C values (fp16 -> fp32) and column indices into UB.
    __aicore__ inline void LoadRowStage(uint64_t rowOff, uint32_t rowNnz,
                                        AscendC::LocalTensor<float> &csrIn,
                                        AscendC::LocalTensor<int32_t> &colInd,
                                        bool scalarConsumer = true, bool needColumns = true) {
        if (betaValue != 0.0f) {
            AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
            AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(rowNnz * sizeof(half)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, static_cast<half>(0.0f)};
            AscendC::DataCopyPad(hBuf, csrValuesGm[rowOff], p, pad);
            SyncMte2ToV();
            AscendC::Cast<float, half>(csrIn, hBuf, AscendC::RoundMode::CAST_NONE, rowNnz);
            SyncVToMte2();
            if (scalarConsumer) {
                SyncVToS();
            }
        }

        if (needColumns) {
            AscendC::DataCopyExtParams cp{1, rowNnz * (uint32_t)sizeof(int32_t), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> cpd{false, 0, 0, 0};
            AscendC::DataCopyPad(colInd, colIndicesGm[rowOff], cp, cpd);
            SyncMte2ToS();
        }
    }

    // Load the A row (or column) into UB once; returns the cache mode used.
    __aicore__ inline int LoadARow16(uint64_t aOff, uint32_t aStride) {
        if (alphaValue == 0.0f) {
            return kACacheNone;
        }
        if (K <= kBlockLen) {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            Gather16(a, matAGm, aOff, aStride, K);
            return kACacheSingle;
        }
        return kACacheNone;
    }

    __aicore__ inline void AcquireScalarOutput() {
        // A preceding vector row/segment may still be reading this queue slot
        // in its output Cast. Only the V->S transition needs this fence.
        if (vectorOutputPending) {
            SyncVToS();
            vectorOutputPending = false;
        }
    }

    // Scalar results are already rounded to half; write their exact bytes
    // without a second vector conversion or shared half staging buffer.
    __aicore__ inline void StoreScalarHalf(uint64_t off, uint32_t count,
                                           AscendC::LocalTensor<half> &values) {
        SyncSToMte3();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[off], values, params);
        SyncMte3ToS();
    }

    // Cast the fp32 results back to fp16 (with saturation) and write them out.
    __aicore__ inline void StoreRow16(uint64_t rowOff, uint32_t rowNnz,
                                      AscendC::LocalTensor<float> &yOut,
                                      bool scalarProducer = true) {
        AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
        if (scalarProducer) {
            SyncSToV();
        }
        AscendC::Cast<half, float>(hBuf, yOut, AscendC::RoundMode::CAST_RINT, rowNnz);
        SyncVToMte3();
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(rowNnz * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(csrValuesGm[rowOff], hBuf, p);
        SyncMte3ToMte2();
        SyncMte3ToV();
        vectorOutputPending = true;
    }

    __aicore__ inline void ProcessRow(uint32_t row) {
        const int32_t start = rowOffsetsGm.GetValue(row);
        const int32_t end = rowOffsetsGm.GetValue(row + 1);
        const uint32_t rowNnz = static_cast<uint32_t>(end - start);
        if (rowNnz == 0) {
            return;
        }
        const uint64_t rowOff = static_cast<uint64_t>(start);
        const uint64_t aOff = aContig ? (static_cast<uint64_t>(row) * ldx) : static_cast<uint64_t>(row);
        const uint32_t aStride = aContig ? 1u : ldx;

        const int aCache = LoadARow16(aOff, aStride);
        const uint32_t segCap = (alphaValue == 0.0f || (K <= kBlockLen && rowNnz >= kVectorMinNnz))
                                    ? kVectorSegmentCap : kOutStageCap;
        for (uint32_t segBase = 0; segBase < rowNnz; segBase += segCap) {
            const uint32_t segNnz = MinU32(rowNnz - segBase, segCap);
            if (segCap == kVectorSegmentCap) {
                ProcessVectorSegment(rowOff + segBase, segNnz, aOff, aStride, aCache);
            } else {
                ProcessRowSegment(rowOff + segBase, segNnz, aOff, aStride, aCache);
            }
        }
    }

    __aicore__ inline void InitDotOffsets(uint32_t count) {
        const uint32_t needed = AlignUp(count, kElemPerRep);
        if (needed <= dotOffsetsCount) {
            return;
        }
        AscendC::LocalTensor<int32_t> offsets = dotOffsetBuf.Get<int32_t>();
        AscendC::CreateVecIndex<int32_t>(offsets, 0, needed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(offsets, offsets, static_cast<int32_t>(kDmaAlignBytes),
                      static_cast<int32_t>(needed));
        AscendC::PipeBarrier<PIPE_V>();
        dotOffsetsCount = needed;
    }

    // Preserve NaNs while applying the same finite/Inf saturation as Saturate.
    __aicore__ inline void SaturateVector(AscendC::LocalTensor<float> &values, uint32_t count) {
        const uint32_t aligned = AlignUp(count, kElemPerRep);
        AscendC::LocalTensor<float> limits = dotStageBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> mask = clampMaskBuf.Get<uint8_t>();
        AscendC::Duplicate(limits, FP16_MAX_VALUE, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(mask, values, limits, AscendC::CMPMODE::GT, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(values, mask, limits, values, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(limits, -FP16_MAX_VALUE, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(mask, values, limits, AscendC::CMPMODE::LT, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(values, mask, limits, values, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessVectorSegment(uint64_t segOff, uint32_t segNnz,
                                                uint64_t aOff, uint32_t aStride, int aCache) {
        if (alphaValue != 0.0f) {
            InitDotOffsets(segNnz);
        }
        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        LoadRowStage(segOff, segNnz, csrIn, colInd, false, alphaValue != 0.0f);
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();
        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> dots = dotStageBuf.Get<float>();
        if (alphaValue == 0.0f) {
            AscendC::Duplicate(yStage, 0.0f, AlignUp(segNnz, kAlign));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
            AscendC::LocalTensor<float> b = bColBuf.Get<float>();
            AscendC::LocalTensor<float> product = mulBuf.Get<float>();
            AscendC::LocalTensor<float> temp = reduceTmpBuf.Get<float>();
            for (uint32_t j = 0; j < segNnz; ++j) {
                const int32_t col = colInd.GetValue(j);
                AscendC::LocalTensor<float> result = dots[j * kAlign];
                if (col < 0 || static_cast<uint32_t>(col) >= colCount) {
                    AscendC::Duplicate(result, 0.0f, kAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                    continue;
                }
                const uint64_t bOff = bContig ? static_cast<uint64_t>(col) * ldy
                                              : static_cast<uint64_t>(col);
                // The reverse fence only protects half staging. The next DMA
                // can overlap this dot's float vector computation.
                Gather16(b, matBGm, bOff, bContig ? 1u : ldy, K);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, a, b, K);
                AscendC::PipeBarrier<PIPE_V>();
                ReduceDot16(result, product, temp);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::LocalTensor<uint32_t> offsets = dotOffsetBuf.Get<int32_t>().ReinterpretCast<uint32_t>();
            AscendC::Gather<float>(yStage, dots, offsets, 0, segNnz);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yStage, yStage, alphaValue, segNnz);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (betaValue != 0.0f) {
            AscendC::Axpy<float, float>(yStage, csrStage, betaValue, segNnz);
            AscendC::PipeBarrier<PIPE_V>();
        }
        SaturateVector(yStage, segNnz);
        inQueueCsr.FreeTensor(csrStage);
        StoreRow16(segOff, segNnz, yStage, false);
        outQueueY.FreeTensor(yStage);
    }

    // Process one segment (up to kOutStageCap nonzeros) of a row.
    __aicore__ inline void ProcessRowSegment(uint64_t segOff, uint32_t segNnz,
                                              uint64_t aOff, uint32_t aStride, int aCache) {
        if (alphaValue == 0.0f || (K <= kBlockLen && segNnz >= kVectorMinNnz)) {
            ProcessVectorSegment(segOff, segNnz, aOff, aStride, aCache);
            return;
        }
        AscendC::LocalTensor<float> csrIn = inQueueCsr.AllocTensor<float>();
        AscendC::LocalTensor<int32_t> colInd = colIndBuf.Get<int32_t>();
        LoadRowStage(segOff, segNnz, csrIn, colInd);
        inQueueCsr.EnQue(csrIn);
        AscendC::LocalTensor<float> csrStage = inQueueCsr.DeQue<float>();

        AscendC::LocalTensor<float> yStage = outQueueY.AllocTensor<float>();
        AcquireScalarOutput();
        AscendC::LocalTensor<half> yHalf = yStage.ReinterpretCast<half>();
        for (uint32_t i = 0; i < segNnz; ++i) {
            const int32_t col = colInd.GetValue(i);
            float dot = 0.0f;
            if (alphaValue != 0.0f && col >= 0 && static_cast<uint32_t>(col) < colCount) {
                const uint64_t bOff = bContig ? (static_cast<uint64_t>(col) * ldy)
                                              : static_cast<uint64_t>(col);
                dot = DotProduct(aOff, aStride, bOff, bContig ? 1u : ldy, aCache);
            }
            const float cv = (betaValue != 0.0f) ? csrStage.GetValue(i) : 0.0f;
            yHalf.SetValue(i, static_cast<half>(Saturate(alphaValue * dot + betaValue * cv)));
        }
        inQueueCsr.FreeTensor(csrStage);

        StoreScalarHalf(segOff, segNnz, yHalf);
        outQueueY.FreeTensor(yStage);
    }

    static constexpr uint32_t kFp16StrideChunk = 128;
    static constexpr uint32_t kVectorSegmentCap = 512;
    static constexpr uint32_t kVectorMinNnz = 8;
    uint32_t dotOffsetsCount = 0;
    bool vectorOutputPending = false;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dotStageBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dotOffsetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> clampMaskBuf;
    AscendC::TPipe *pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFullBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colIndBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridePadBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> strideOffsetBuf;
    bool needStrideGather = false;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueCsr;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<half> matAGm;
    AscendC::GlobalTensor<half> matBGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::GlobalTensor<int32_t> colIndicesGm;
    AscendC::GlobalTensor<half> csrValuesGm;
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
    bool aContig = true;
    bool bContig = true;
    uint32_t nnzPartition = 0;
    uint32_t nnzPerCore = 0;
    uint32_t remainderNnz = 0;
    uint32_t ldx = 0;
    uint32_t ldy = 0;
    uint32_t orderXCol = 0;
    uint32_t orderYCol = 0;
};

#endif // SDDMM_ARCH22_FP16_KERNEL_H_
