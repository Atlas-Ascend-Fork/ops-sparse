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
#include "cube_spmm.h"

using namespace AscendC;

namespace cube_spmm {

template <typename aType, typename bType, typename cType, typename idxType, bool outputPrecleared = false>
class CubeSpmmKernel {
    static constexpr uint32_t CUBE_BLOCK_K = static_cast<uint32_t>(kTileK);

public:
    __aicore__ inline CubeSpmmKernel() {}

    __aicore__ inline void Init(
        GM_ADDR rw_ptr, GM_ADDR col_ref, GM_ADDR vals,
        GM_ADDR b, GM_ADDR core_info, GM_ADDR c,
        GM_ADDR workspaceGM, const CubeSpmmTilingData &tiling)
    {
        // workspaceGM is kept as a formal kernel parameter (same shape as the
        // official spmm kernel) for future scratch use; currently unused.
        this->M = tiling.M;
        this->K = tiling.K;
        this->N = tiling.N;
        this->tileM = tiling.tileM;
        this->tileN = tiling.tileN;
        this->tailM = tiling.tailM;
        this->lastKLength = tiling.lastKLength;
        this->bLd = tiling.bLd;
        this->cLd = tiling.cLd;

        // Current kernel hardcodes tileM/tileN to 16; reject any mismatch.
        if (this->tileM != 16 || this->tileN != 16) {
            this->rowWindowNum = 0;
            return;
        }
        this->CUBE_BLOCK_M = static_cast<uint32_t>(this->tileM);
        this->CUBE_BLOCK_N = static_cast<uint32_t>(this->tileN);
        this->CUBE_BLOCK_SIZE = this->CUBE_BLOCK_M * CUBE_BLOCK_K;

        uint32_t usedCoreNum = tiling.usedCoreNum;

        AscendC::GlobalTensor<int32_t> coreInfoGm;
        coreInfoGm.SetGlobalBuffer((__gm__ int32_t *)core_info,
                                   static_cast<uint64_t>(usedCoreNum) * 4);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        int64_t infoBase = static_cast<int64_t>(blockIdx) * 4;
        this->rwStart = coreInfoGm.GetValue(infoBase);
        this->rwEnd = coreInfoGm.GetValue(infoBase + 1);
        this->blkStart = coreInfoGm.GetValue(infoBase + 2);
        this->blkEnd = coreInfoGm.GetValue(infoBase + 3);
        this->rowWindowNum = rwEnd - rwStart;

        if (this->rowWindowNum <= 0) {
            return;
        }

        this->lastRowWindow = (this->M + this->tileM - 1) / this->tileM - 1;

        rwPtrGm.SetGlobalBuffer((__gm__ int64_t *)rw_ptr + rwStart,
                                static_cast<uint64_t>(this->rowWindowNum) + 1);

        cGm.SetGlobalBuffer((__gm__ cType *)c + (uint64_t)rwStart * CUBE_BLOCK_M * cLd,
                            (uint64_t)this->rowWindowNum * CUBE_BLOCK_M * cLd);

        // N-direction tiling: local buffers are sized for at most kNChunkSize columns.
        this->nProcTile = (this->N < cube_spmm::kNChunkSize) ? this->N : cube_spmm::kNChunkSize;

        // Zero-initialize the output GM region belonging to this core so that
        // stale host data is not picked up by atomic-add paths at row-window
        // boundaries. The zero buffer is chunk-sized; C is zeroed one N-chunk at
        // a time before the main compute loop begins.
        if constexpr (!outputPrecleared) {
            pipe.InitBuffer(zeroBuf, static_cast<uint32_t>(CUBE_BLOCK_M * nProcTile * sizeof(cType)));
            AscendC::LocalTensor<cType> cZero = zeroBuf.Get<cType>();
            FillLocalBufferWithZero(cZero, static_cast<int64_t>(CUBE_BLOCK_M) * nProcTile);

            for (int64_t nOffset = 0; nOffset < N; nOffset += nProcTile) {
                int64_t chunkN = (nOffset + nProcTile <= N) ? nProcTile : (N - nOffset);
                for (int32_t row = 0; row < this->rowWindowNum; row++) {
                    bool isLastGlobalRow = (this->rwStart + row == this->lastRowWindow);
                    uint32_t rowsToWrite = isLastGlobalRow ? static_cast<uint32_t>(this->tailM) : this->CUBE_BLOCK_M;
                    auto cGmRow = this->cGm[static_cast<int64_t>(row) * CUBE_BLOCK_M * cLd + nOffset];
                    CopyZeroRows(cGmRow, cZero, rowsToWrite, chunkN, cLd);
                }
            }
        }

        colRefGm.SetGlobalBuffer((__gm__ int32_t *)col_ref + rwPtrGm.GetValue(0) * CUBE_BLOCK_K,
                                 (rwPtrGm.GetValue(this->rowWindowNum) - rwPtrGm.GetValue(0)) * CUBE_BLOCK_K);
        valsGm.SetGlobalBuffer((__gm__ aType *)vals + CUBE_BLOCK_SIZE * rwPtrGm.GetValue(0),
                               CUBE_BLOCK_SIZE * (rwPtrGm.GetValue(this->rowWindowNum) - rwPtrGm.GetValue(0)));

        bGm.SetGlobalBuffer((__gm__ bType *)b, (uint64_t)K * bLd);

        pipe.InitBuffer(inQueueA1, 2, CUBE_BLOCK_SIZE * sizeof(aType));
        pipe.InitBuffer(inQueueA2, 2, CUBE_BLOCK_SIZE * sizeof(aType));
        pipe.InitBuffer(inQueueB1, 2, CUBE_BLOCK_K * nProcTile * sizeof(bType));
        pipe.InitBuffer(inQueueB2, 2, CUBE_BLOCK_K * nProcTile * sizeof(bType));
        pipe.InitBuffer(outQueueCO1, 1, CUBE_BLOCK_M * nProcTile * sizeof(cType));
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = 0; row < rowWindowNum; row++) {
            int64_t totalBlocks = rwPtrGm.GetValue(row + 1) - rwPtrGm.GetValue(row);

            int64_t curStart = 0;
            int64_t curEnd = totalBlocks;
            if (row == 0) {
                curStart = blkStart;
            }
            if (row == rowWindowNum - 1) {
                curEnd = blkEnd;
            }

            if (curEnd <= curStart) {
                continue;
            }

            bool needAtomic = false;
            if (row == 0 && blkStart > 0) { needAtomic = true; }
            if (row == rowWindowNum - 1 && blkEnd < totalBlocks) { needAtomic = true; }

            for (int64_t nOffset = 0; nOffset < N; nOffset += nProcTile) {
                int64_t chunkN = (nOffset + nProcTile <= N) ? nProcTile : (N - nOffset);

                AscendC::LocalTensor<cType> cAcc = outQueueCO1.AllocTensor<cType>();

                bool first = true;
                for (int64_t i = curStart; i < curEnd; i++) {
                    CopyInA(row, i);
                    SplitA();
                    CopyInB(row, i, nOffset, chunkN);
                    SplitB(chunkN);
                    ComputeAccOnCO1(cAcc, first, chunkN);
                    first = false;
                }
                outQueueCO1.EnQue<cType>(cAcc);
                CopyOut(row, nOffset, chunkN, needAtomic);
                outQueueCO1.FreeTensor(cAcc);
            }
        }
    }

private:
    __aicore__ inline void CopyInA(int64_t row, int64_t i)
    {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
        int64_t blockOffset = rwPtrGm.GetValue(row) - rwPtrGm.GetValue(0) + i;
        auto aGm = this->valsGm[blockOffset * CUBE_BLOCK_SIZE];
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = CUBE_BLOCK_M;
        params.dValue = CUBE_BLOCK_K;
        params.srcNdMatrixStride = 0;
        params.srcDValue = CUBE_BLOCK_K;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;

        AscendC::DataCopy(a1Local, aGm, params);
        inQueueA1.EnQue<aType>(a1Local);
    }

    __aicore__ inline void CopyZeroRows(
        const AscendC::GlobalTensor<cType> &cGmRow,
        AscendC::LocalTensor<cType> &cZero,
        uint32_t rowsToWrite,
        int64_t cols,
        int64_t ld)
    {
        AscendC::DataCopyParams zeroCopyParams;
        zeroCopyParams.blockCount = 1;
        zeroCopyParams.blockLen = static_cast<uint16_t>(CUBE_BLOCK_M * nProcTile * sizeof(cType) / 32);
        zeroCopyParams.srcStride = 0;
        zeroCopyParams.dstStride = 0;

        constexpr uint32_t kMaxDataCopyStride = static_cast<uint32_t>(UINT16_MAX);
        const uint32_t ldStride32B = static_cast<uint32_t>(ld - cols) * sizeof(cType) / 32;

        if (ld == cols) {
            zeroCopyParams.blockCount = 1;
            zeroCopyParams.blockLen = static_cast<uint16_t>(rowsToWrite * cols * sizeof(cType) / 32);
            zeroCopyParams.dstStride = 0;
            AscendC::DataCopy(cGmRow, cZero, zeroCopyParams);
        } else if (ldStride32B <= kMaxDataCopyStride) {
            zeroCopyParams.blockCount = rowsToWrite;
            zeroCopyParams.blockLen = static_cast<uint16_t>(cols * sizeof(cType) / 32);
            zeroCopyParams.dstStride = static_cast<uint16_t>(ldStride32B);
            AscendC::DataCopy(cGmRow, cZero, zeroCopyParams);
        } else {
            zeroCopyParams.blockCount = 1;
            zeroCopyParams.blockLen = static_cast<uint16_t>(cols * sizeof(cType) / 32);
            zeroCopyParams.dstStride = 0;
            for (uint32_t r = 0; r < rowsToWrite; ++r) {
                auto cGmRowR = cGmRow[r * ld];
                AscendC::DataCopy(cGmRowR, cZero, zeroCopyParams);
            }
        }
    }

    // L1 Fill counts 32-byte blocks, not elements. Order its MTE2 writes
    // before the subsequent L1-to-GM copies on MTE3.
    __aicore__ inline void FillLocalBufferWithZero(
        AscendC::LocalTensor<cType> buf,
        int64_t totalElements)
    {
        AscendC::InitConstValueParams<cType> params;
        params.repeatTimes = 1;
        params.blockNum = static_cast<uint16_t>(totalElements * sizeof(cType) / 32);
        params.dstGap = 0;
        params.initValue = static_cast<cType>(0);

        AscendC::Fill(buf, params);
        const auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(event);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(event);
    }

    __aicore__ inline void CopyInB(int64_t row, int64_t i, int64_t nOffset, int64_t chunkN)
    {
        AscendC::LocalTensor<bType> b1local = inQueueB1.AllocTensor<bType>();
        AscendC::DataCopyParams b1param;
        b1param.blockCount = static_cast<uint16_t>(chunkN / CUBE_BLOCK_N);
        b1param.blockLen = CUBE_BLOCK_N * sizeof(bType) / 32;
        b1param.srcStride = 0;
        b1param.dstStride = (CUBE_BLOCK_K - 1) * CUBE_BLOCK_N * sizeof(bType) / 32;
        for (int j = 0; j < CUBE_BLOCK_K; ++j) {
            int64_t colRefIdx = (rwPtrGm(row) - rwPtrGm(0) + i) * CUBE_BLOCK_K + j;
            int64_t row_index = colRefGm.GetValue(colRefIdx);
            DataCopy(b1local[static_cast<int64_t>(j) * CUBE_BLOCK_N],
                     bGm[row_index * bLd + nOffset],
                     b1param);
        }
        inQueueB1.EnQue<bType>(b1local);
    }

    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();

        AscendC::LoadData2DParams params;
        params.repeatTimes = 1;
        params.srcStride = 0;
        params.dstGap = 0;
        params.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, params);

        inQueueA2.EnQue<aType>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitB(int64_t chunkN)
    {
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();

        // LoadData repeatTimes is uint8, so each call can handle at most 255 tile
        // columns. Process the chunk in batches of up to 255 tile columns to keep
        // efficiency while staying within the hardware limit.
        AscendC::LoadData2DParams loadDataparams;
        loadDataparams.srcStride = 1;
        loadDataparams.dstGap = 0;
        loadDataparams.ifTranspose = true;

        constexpr int64_t kMaxLoadDataRepeats = 255;
        constexpr int64_t kTileBytes =
            static_cast<int64_t>(kTileK) * static_cast<int64_t>(kTileN) * static_cast<int64_t>(sizeof(bType));
        constexpr int64_t kTileElems = kTileBytes / static_cast<int64_t>(sizeof(bType));
        int64_t numTiles = chunkN / kTileN;
        int64_t processedTiles = 0;
        while (processedTiles < numTiles) {
            int64_t batchTiles = numTiles - processedTiles;
            if (batchTiles > kMaxLoadDataRepeats) {
                batchTiles = kMaxLoadDataRepeats;
            }
            loadDataparams.repeatTimes = static_cast<uint16_t>(batchTiles);
            AscendC::LoadData(b2Local[processedTiles * kTileElems],
                              b1Local[processedTiles * kTileElems],
                              loadDataparams);
            processedTiles += batchTiles;
        }

        inQueueB2.EnQue<bType>(b2Local);
        inQueueB1.FreeTensor(b1Local);
    }

    __aicore__ inline void ComputeAccOnCO1(AscendC::LocalTensor<cType> &cAcc, bool first, int64_t chunkN)
    {
        AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
        AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();

        AscendC::MmadParams p{};
        p.m = CUBE_BLOCK_M;
        p.k = CUBE_BLOCK_K;
        p.n = static_cast<int32_t>(chunkN);

        if (first) {
            p.cmatrixInitVal = true;
            AscendC::Mmad(cAcc, a2Local, b2Local, p);
        } else {
            p.cmatrixInitVal = false;
            AscendC::Mmad(cAcc, a2Local, b2Local, cAcc, p);
        }

        AscendC::PipeBarrier<PIPE_M>();

        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(int64_t row, int64_t nOffset, int64_t chunkN, bool needAtomic)
    {
        auto cGm = this->cGm[row * CUBE_BLOCK_M * cLd + nOffset];
        AscendC::LocalTensor<cType> c1Local = outQueueCO1.DeQue<cType>();

        bool isLastGlobalRow = (this->rwStart + row == this->lastRowWindow);
        uint32_t mSize = isLastGlobalRow ? static_cast<uint32_t>(this->tailM) : this->CUBE_BLOCK_M;

        AscendC::FixpipeParamsV220 params;
        params.ndNum = 1;
        params.mSize = mSize;
        params.nSize = static_cast<uint16_t>(chunkN);
        params.srcStride = CUBE_BLOCK_M;
        params.dstStride = cLd;
        params.srcNdStride = 0;
        params.dstNdStride = 0;

        if (needAtomic) {
            AscendC::SetAtomicAdd<cType>();
        }
        AscendC::Fixpipe(cGm, c1Local, params);
        if (needAtomic) {
            AscendC::SetAtomicNone();
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    // Temporary L1 buffer used only in Init to zero-initialize the output GM.
    AscendC::TBuf<AscendC::TPosition::A1> zeroBuf;

    AscendC::GlobalTensor<int64_t> rwPtrGm;
    AscendC::GlobalTensor<int32_t> colRefGm;
    AscendC::GlobalTensor<aType> valsGm;

    AscendC::GlobalTensor<bType> bGm;
    AscendC::GlobalTensor<cType> cGm;

    int64_t M = 0;
    int64_t K = 0;
    int64_t N = 0;
    int32_t tileM = 0;
    int32_t tileN = 0;
    int32_t nProcTile = 0;
    int32_t tailM = 0;
    int32_t bLd = 0;
    int32_t cLd = 0;
    int32_t rwStart = 0;
    int32_t rwEnd = 0;
    int32_t blkStart = 0;
    int32_t blkEnd = 0;
    int64_t lastRowWindow = 0;
    int64_t rowWindowNum = 0;
    uint32_t lastKLength = 0;

    // Tile sizes are read from TilingData and validated against the hardcoded
    // kernel assumption of 16; CUBE_BLOCK_* are derived from them.
    uint32_t CUBE_BLOCK_M = 0;
    uint32_t CUBE_BLOCK_N = 0;
    uint32_t CUBE_BLOCK_SIZE = 0;
};

}  // namespace cube_spmm

extern "C" __global__ __aicore__ void cube_spmm_kernel(
    GM_ADDR rw_ptr, GM_ADDR col_ref, GM_ADDR vals,
    GM_ADDR b, GM_ADDR core_info, GM_ADDR c,
    GM_ADDR workspaceGM, const cube_spmm::CubeSpmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cube_spmm::CubeSpmmKernel<half, half, float, int32_t> op;
    op.Init(rw_ptr, col_ref, vals, b, core_info, c, workspaceGM, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void cube_spmm_precleared_kernel(
    GM_ADDR rw_ptr, GM_ADDR col_ref, GM_ADDR vals,
    GM_ADDR b, GM_ADDR core_info, GM_ADDR c,
    GM_ADDR workspaceGM, const cube_spmm::CubeSpmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cube_spmm::CubeSpmmKernel<half, half, float, int32_t, true> op;
    op.Init(rw_ptr, col_ref, vals, b, core_info, c, workspaceGM, tiling);
    op.Process();
}

extern "C" void cube_spmm_kernel_launch(
    GM_ADDR rw_ptr, GM_ADDR col_ref, GM_ADDR vals,
    GM_ADDR b, GM_ADDR core_info, GM_ADDR c,
    GM_ADDR workspaceGM, const cube_spmm::CubeSpmmTilingData &tiling,
    uint32_t usedCoreNum, void *stream, bool outputPrecleared)
{
    if (outputPrecleared) {
        cube_spmm_precleared_kernel<<<usedCoreNum, nullptr, stream>>>(
            rw_ptr, col_ref, vals,
            b, core_info, c,
            workspaceGM, tiling);
    } else {
        cube_spmm_kernel<<<usedCoreNum, nullptr, stream>>>(
            rw_ptr, col_ref, vals,
            b, core_info, c,
            workspaceGM, tiling);
    }
}
