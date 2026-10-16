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
#include "spmm.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = SPMM_ARCH22_BUFFER_NUM;

class SpmmArch22Kernel {
public:
    __aicore__ inline void Init(
        GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
        GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tilingGm)
    {
        __gm__ SpmmArch22TilingData *td =
            reinterpret_cast<__gm__ SpmmArch22TilingData *>(tilingGm);

        M = td->M;
        K = td->K;
        N = td->N;
        nnz = td->nnz;
        alphaVal = td->alphaHost;
        betaVal  = td->betaHost;
        skipBeta = (betaVal == 0.0f);
        nChunks  = td->nChunks;

        uint32_t blockDim = GetBlockNum();
        uint32_t blockId  = GetBlockIdx();

        __gm__ int32_t *binEdge =
            reinterpret_cast<__gm__ int32_t *>(workspace + td->binEdgeOffset);
        myRowStart = static_cast<uint32_t>(binEdge[blockId]);
        myRowEnd   = static_cast<uint32_t>(binEdge[blockId + 1]);

        reorderGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workspace + td->reorderOffset), M);
        rowOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowOff), M + 1);
        colIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(colIdx), nnz);
        valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values), nnz);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matB), static_cast<uint64_t>(K) * N);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC), static_cast<uint64_t>(M) * N);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC), static_cast<uint64_t>(M) * N);

        uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
        uint64_t nMax = availUB / (4 * sizeof(float));
        nMax = (nMax / SPMM_ARCH22_ALIGN) * SPMM_ARCH22_ALIGN;
        if (nMax > N)
        {
            nMax = N;
        }
        nMaxPerChunk = static_cast<uint32_t>(nMax);
        nAlignChunk = ROUNDUP(nMaxPerChunk, SPMM_ARCH22_ALIGN);
        uint32_t rowBytes = nAlignChunk * sizeof(float);

        pipe.InitBuffer(bQueue, BUFFER_NUM, rowBytes);
        pipe.InitBuffer(outQueue, 1, rowBytes);
        pipe.InitBuffer(tmpBuf, rowBytes);
    }

    __aicore__ inline void Process()
    {
        if (myRowStart >= myRowEnd) { return; }

        for (uint32_t chunkIdx = 0; chunkIdx < nChunks; chunkIdx++) {
            uint32_t colStart = chunkIdx * nMaxPerChunk;
            uint32_t colEnd = colStart + nMaxPerChunk;
            if (colEnd > N) { colEnd = N; }
            uint32_t chunkLen = colEnd - colStart;
            uint32_t chunkAlign = ROUNDUP(chunkLen, SPMM_ARCH22_ALIGN);

            ProcessChunk(colStart, chunkLen, chunkAlign);
        }
    }

private:
    __aicore__ inline void ProcessChunk(
        uint32_t colStart, uint32_t chunkLen, uint32_t chunkAlign)
    {
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        DataCopyParams copyParams = {1, (uint16_t)(chunkLen * sizeof(float)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        for (uint32_t ri = myRowStart; ri < myRowEnd; ri++) {
            int32_t row = reorderGm.GetValue(ri);
            uint32_t rs = static_cast<uint32_t>(rowOffsetsGm.GetValue(row));
            uint32_t re = static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1));
            uint64_t outBase = static_cast<uint64_t>(row) * N + colStart;

            LocalTensor<float> accLocal = outQueue.AllocTensor<float>();
            Duplicate(accLocal, 0.0f, chunkAlign);

            if (rs < re) {
                uint32_t col0 = static_cast<uint32_t>(colIndicesGm.GetValue(rs));
                LocalTensor<float> bFirst = bQueue.AllocTensor<float>();
                DataCopyPad(bFirst, bGm[static_cast<uint64_t>(col0) * N + colStart], copyParams, padParams);
                bQueue.EnQue(bFirst);

                for (uint32_t idx = rs; idx < re; idx++) {
                    LocalTensor<float> bCur = bQueue.DeQue<float>();
                    float val = valuesGm.GetValue(idx) * alphaVal;

                    if (idx + 1 < re) {
                        uint32_t nextCol = static_cast<uint32_t>(colIndicesGm.GetValue(idx + 1));
                        LocalTensor<float> bNext = bQueue.AllocTensor<float>();
                        DataCopyPad(bNext, bGm[static_cast<uint64_t>(nextCol) * N + colStart], copyParams, padParams);
                        bQueue.EnQue(bNext);
                    }

                    Muls(tmp, bCur, val, chunkAlign);
                    Add(accLocal, accLocal, tmp, chunkAlign);
                    bQueue.FreeTensor(bCur);
                }
            }

            if (!skipBeta) {
                LocalTensor<float> cLocal = bQueue.AllocTensor<float>();
                DataCopyPad(cLocal, cGm[outBase], copyParams, padParams);
                bQueue.EnQue(cLocal);
                cLocal = bQueue.DeQue<float>();
                Muls(tmp, cLocal, betaVal, chunkAlign);
                Add(accLocal, accLocal, tmp, chunkAlign);
                bQueue.FreeTensor(cLocal);
            }

            outQueue.EnQue(accLocal);
            accLocal = outQueue.DeQue<float>();
            DataCopyPad(yGm[outBase], accLocal, copyParams);
            outQueue.FreeTensor(accLocal);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> bQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<int32_t> reorderGm;
    GlobalTensor<int32_t> rowOffsetsGm;
    GlobalTensor<int32_t> colIndicesGm;
    GlobalTensor<float>   valuesGm;
    GlobalTensor<float>   bGm;
    GlobalTensor<float>   cGm;
    GlobalTensor<float>   yGm;

    uint32_t M, K, N, nnz;
    uint32_t nChunks, nMaxPerChunk, nAlignChunk;
    uint32_t myRowStart, myRowEnd;
    float alphaVal, betaVal;
    bool skipBeta;
};

__global__ __aicore__ void spmm_arch22_fp32(
    GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
    GM_ADDR matB, GM_ADDR matC,
    GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmmArch22Kernel op;
    op.Init(rowOff, colIdx, values, matB, matC, workspace, tilingGm);
    op.Process();
}

void spmm_arch22_kernel_launch(
    GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tiling,
    uint32_t blockDim, void *stream)
{
    spmm_arch22_fp32<<<blockDim, nullptr, stream>>>(
        rowOff, colIdx, values,
        matB, matC,
        workspace, tiling);
}
