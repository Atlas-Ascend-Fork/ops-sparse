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

/*
 * Preserve master dtype/base support while retaining the synchronized batched
 * write path. T is the storage view; complex64 uses two float words per element.
 * Vector operations validate indices only; values move without arithmetic.
 */

#include "kernel_operator.h"
#include "scatter.h"
#include "scatter_kernel.h"
#include <cstdio>

using namespace AscendC;

// A scalar-written value must occupy an independent 32-byte source slot for
// DataCopyPad.  Batch those slots so one producer/consumer event pair protects
// many sparse singleton writes.
constexpr uint32_t SCATTER_SCALAR_SLOT_BYTES = 32;
constexpr uint32_t SCATTER_SCALAR_BATCH = 128;
constexpr uint32_t SCATTER_CONTIG_VECTOR_MIN = 256;

template <typename T, uint32_t ELEM_BYTES>
class KernelScatter {
public:
    __aicore__ inline KernelScatter()
    {}

    __aicore__ inline void Init(GM_ADDR valGmAddr, GM_ADDR idxGmAddr,
                                GM_ADDR yGmAddr, const ScatterTilingData &tiling)
    {
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        if (blockIdx >= tiling.blockNum) {
            return;
        }
        idxBase_ = tiling.idxBase;
        coreNnz_   = tiling.coreNnzCount[blockIdx];
        coreOffset_ = tiling.coreNnzOffset[blockIdx];

        valGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(valGmAddr),
                              static_cast<uint64_t>(tiling.nnz) * GM_STRIDE);
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxGmAddr), tiling.nnz);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yGmAddr), tiling.yLen / sizeof(T));

        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNn = tiling.tileNn;
        uint32_t indexAligned = ((tileNn + 7) / 8) * 8;
        pipe.InitBuffer(inQueueVal, SCATTER_BUFFER_NUM, tiling.tileNnAligned * ELEM_BYTES);
        pipe.InitBuffer(inQueueIdx, SCATTER_BUFFER_NUM, indexAligned * sizeof(int32_t));
        // Cover both the largest run and 128 independent 32-byte staging slots.
        uint32_t outputBytes = tiling.tileNnAligned * ELEM_BYTES;
        constexpr uint32_t stageBytes = SCATTER_SCALAR_BATCH * SCATTER_SCALAR_SLOT_BYTES;
        if (outputBytes < stageBytes) {
            outputBytes = stageBytes;
        }
        pipe.InitBuffer(outQueue, 1, outputBytes);
        contigCheckEnabled_ = tileNn >= SCATTER_CONTIG_VECTOR_MIN;
        if (contigCheckEnabled_) {
            pipe.InitBuffer(contigCheckBuf_, indexAligned * sizeof(int32_t));
        }

        tileNn_ = tileNn;
    }

    __aicore__ inline void Process()
    {
        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNum = (coreNnz_ + tileNn_ - 1) / tileNn_;

        {
            uint32_t curTileNn = (coreNnz_ < tileNn_) ? coreNnz_ : tileNn_;
            CopyInAsync(coreOffset_, curTileNn);
        }

        for (uint32_t tile = 0; tile < tileNum; tile++) {
            uint32_t localOffset = tile * tileNn_;
            if (localOffset >= coreNnz_) {
                break;
            }
            uint32_t curTileNn = coreNnz_ - localOffset;
            if (curTileNn > tileNn_) {
                curTileNn = tileNn_;
            }

            WaitCopyIn();

            if (tile + 1 < tileNum) {
                uint32_t nextOffset = coreOffset_ + (tile + 1) * tileNn_;
                uint32_t nextNn     = coreNnz_ - (tile + 1) * tileNn_;
                if (nextNn > tileNn_) {
                    nextNn = tileNn_;
                }
                CopyInAsync(nextOffset, nextNn);
            }

            Compute(curTileNn, coreOffset_ + localOffset);
        }
    }

private:
    template <HardEvent Event>
    __aicore__ inline void SyncEvent()
    {
        event_t eventId = static_cast<event_t>(pipe.AllocEventID<Event>());
        SetFlag<Event>(eventId);
        WaitFlag<Event>(eventId);
        pipe.ReleaseEventID<Event>(eventId);
    }

    __aicore__ inline void CopyInAsync(uint32_t gmOffset, uint32_t curTileNn)
    {
        auto valLocal = inQueueVal.AllocTensor<T>();
        DataCopyPad(valLocal, valGm[static_cast<uint64_t>(gmOffset) * GM_STRIDE],
            {1, static_cast<uint16_t>(curTileNn * ELEM_BYTES), 0, 0},
            {false, 0, 0, 0});
        inQueueVal.EnQue(valLocal);

        auto idxLocal = inQueueIdx.AllocTensor<int32_t>();
        DataCopyPad(idxLocal, idxGm[gmOffset],
            {1, static_cast<uint16_t>(curTileNn * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        inQueueIdx.EnQue(idxLocal);
    }

    __aicore__ inline void WaitCopyIn()
    {
        valBuf_   = inQueueVal.DeQue<T>();
        idxBuf32_ = inQueueIdx.DeQue<int32_t>();
        SyncEvent<HardEvent::MTE2_S>();
    }

    __aicore__ inline void Compute(uint32_t curTileNn, uint32_t tileGmBase)
    {
        auto outLocal = outQueue.AllocTensor<T>();
        uint32_t scalarOffsets[SCATTER_SCALAR_BATCH];
        uint32_t scalarCount = 0;
        if (TryWriteContiguousTile(curTileNn, outLocal)) {
            ReleaseTile(outLocal);
            return;
        }
        uint32_t runStart = 0;
        int32_t prevIdx = idxBuf32_.GetValue(0);

        for (uint32_t i = 1; i <= curTileNn; i++) {
            bool isEnd = (i == curTileNn);
            bool isDisconnected = false;
            int32_t curIdx = 0;

            if (!isEnd) {
                curIdx = idxBuf32_.GetValue(i);
                isDisconnected = (prevIdx == static_cast<int32_t>(0x7fffffff) ||
                                  curIdx != prevIdx + 1);
            }

            if (isEnd || isDisconnected) {
                uint32_t runLen = i - runStart;
                int64_t targetIdx = static_cast<int64_t>(idxBuf32_.GetValue(runStart)) - idxBase_;

                if (runLen == 1) {
                    for (uint32_t j = 0; j < GM_STRIDE; ++j) {
                        outLocal.SetValue(scalarCount * SLOT_T_ELEMENTS + j,
                                          valBuf_.GetValue(runStart * GM_STRIDE + j));
                    }
                    scalarOffsets[scalarCount++] = runStart;
                    if (scalarCount == SCATTER_SCALAR_BATCH) {
                        FlushScalarBatch(outLocal, scalarOffsets, scalarCount);
                        scalarCount = 0;
                    }
                } else {
                    FlushScalarBatch(outLocal, scalarOffsets, scalarCount);
                    scalarCount = 0;
                    WriteMultiRun(outLocal, targetIdx, runStart, runLen, tileGmBase);
                }

                runStart = i;
                if (!isEnd) {
                    prevIdx = curIdx;
                }
            } else {
                prevIdx = curIdx;
            }
        }

        FlushScalarBatch(outLocal, scalarOffsets, scalarCount);
        ReleaseTile(outLocal);
    }

    __aicore__ inline void ReleaseTile(LocalTensor<T> &outLocal)
    {
        outQueue.FreeTensor(outLocal);
        inQueueVal.FreeTensor(valBuf_);
        inQueueIdx.FreeTensor(idxBuf32_);
    }

    __aicore__ inline bool TryWriteContiguousTile(uint32_t curTileNn,
                                                   LocalTensor<T> &outLocal)
    {
        if (!contigCheckEnabled_ || curTileNn < SCATTER_CONTIG_VECTOR_MIN ||
            (idxBuf32_.GetValue(0) < idxBase_)) {
            return false;
        }
        int32_t firstIdx = idxBuf32_.GetValue(0);
        if (firstIdx > static_cast<int32_t>(0x7fffffff - (curTileNn - 1))) {
            return false;
        }
        uint32_t quickCount = curTileNn < 8 ? curTileNn : 8;
        for (uint32_t i = 1; i < quickCount; ++i) {
            if (idxBuf32_.GetValue(i) != firstIdx + static_cast<int32_t>(i)) {
                return false;
            }
        }

        uint32_t vectorCount = curTileNn & ~uint32_t(63);
        auto expected = contigCheckBuf_.Get<int32_t>();
        SyncEvent<HardEvent::MTE2_V>();
        CreateVecIndex<int32_t>(expected, firstIdx, vectorCount);
        PipeBarrier<PIPE_V>();
        Sub(expected, expected, idxBuf32_, vectorCount);
        PipeBarrier<PIPE_V>();
        auto packedMask = outLocal.template ReinterpretCast<uint8_t>();
        Compares<int32_t, uint8_t>(packedMask, expected, 0, CMPMODE::EQ, vectorCount);
        SyncEvent<HardEvent::V_S>();
        auto words = outLocal.template ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < vectorCount / 64; ++i) {
            if (words.GetValue(i) != static_cast<uint64_t>(-1)) {
                return false;
            }
        }
        for (uint32_t i = vectorCount; i < curTileNn; ++i) {
            if (idxBuf32_.GetValue(i) != firstIdx + static_cast<int32_t>(i)) {
                return false;
            }
        }

        SyncEvent<HardEvent::MTE2_MTE3>();
        DataCopyPad(yGm[static_cast<uint64_t>(firstIdx - idxBase_) * GM_STRIDE], valBuf_,
            {1, static_cast<uint32_t>(curTileNn * ELEM_BYTES), 0, 0, 0});
        SyncEvent<HardEvent::MTE3_S>();
        return true;
    }

    __aicore__ inline void FlushScalarBatch(LocalTensor<T> &outLocal,
                                             const uint32_t *scalarOffsets,
                                             uint32_t scalarCount)
    {
        if (scalarCount == 0) {
            return;
        }
        SyncEvent<HardEvent::S_MTE3>();
        for (uint32_t i = 0; i < scalarCount; ++i) {
            int64_t targetIdx = static_cast<int64_t>(idxBuf32_.GetValue(scalarOffsets[i])) - idxBase_;
            DataCopyPad(yGm[static_cast<uint64_t>(targetIdx) * GM_STRIDE], outLocal[i * SLOT_T_ELEMENTS],
                {1, static_cast<uint32_t>(ELEM_BYTES), 0, 0, 0});
        }
        SyncEvent<HardEvent::MTE3_S>();
    }

    __aicore__ inline void WriteMultiRun(LocalTensor<T> &outLocal,
                                          int64_t targetIdx, uint32_t srcOffset, uint32_t runLen,
                                          uint32_t tileGmBase)
    {
        AscendC::DataCopyExtParams cp{1, runLen * ELEM_BYTES, 0, 0, 0};
        if ((srcOffset * ELEM_BYTES & 31) == 0) {
            SyncEvent<HardEvent::MTE2_MTE3>();
            DataCopyPad(yGm[static_cast<uint64_t>(targetIdx) * GM_STRIDE],
                        valBuf_[srcOffset * GM_STRIDE], cp);
        } else {
            // Arbitrary dtype/run alignment: preserve master's byte-copy path
            // through an aligned UB buffer, without vector value conversion.
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(outLocal,
                        valGm[(static_cast<uint64_t>(tileGmBase) + srcOffset) * GM_STRIDE], cp, pad);
            SyncEvent<HardEvent::MTE2_MTE3>();
            DataCopyPad(yGm[static_cast<uint64_t>(targetIdx) * GM_STRIDE], outLocal, cp);
        }
        SyncEvent<HardEvent::MTE3_S>();
    }

private:
    static constexpr uint32_t GM_STRIDE = ELEM_BYTES / sizeof(T);
    static constexpr uint32_t SLOT_T_ELEMENTS = SCATTER_SCALAR_SLOT_BYTES / sizeof(T);
    TPipe pipe;
    TQue<TPosition::VECIN, SCATTER_BUFFER_NUM> inQueueVal;
    TQue<TPosition::VECIN, SCATTER_BUFFER_NUM> inQueueIdx;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> contigCheckBuf_;
    GlobalTensor<T> valGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<T> yGm;
    LocalTensor<T> valBuf_;
    LocalTensor<int32_t> idxBuf32_;
    uint32_t coreNnz_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t tileNn_ = 0;
    bool contigCheckEnabled_ = false;
    int32_t idxBase_ = 0;
};


template <typename T, uint32_t ELEM_BYTES>
__global__ __vector__ void scatter_custom(
    GM_ADDR valGm, GM_ADDR idxGm, GM_ADDR yGm, const ScatterTilingData tiling)
{
    KernelScatter<T, ELEM_BYTES> op;
    op.Init(valGm, idxGm, yGm, tiling);
    op.Process();
}

void scatter_kernel_do(
    GM_ADDR valDev, GM_ADDR idxDev, GM_ADDR yDev, aclDataType valueType,
    const ScatterTilingData &tiling, uint32_t blockNum, void *stream)
{
    // Host 校验链已保证 valueType ∈ 五 dtype 集合，default 不达
    switch (valueType) {
        case ACL_INT8:
            scatter_custom<int8_t, 1><<<blockNum, nullptr, stream>>>(
                valDev, idxDev, yDev, tiling);
            break;
        case ACL_FLOAT16:
            scatter_custom<half, 2><<<blockNum, nullptr, stream>>>(
                valDev, idxDev, yDev, tiling);
            break;
        case ACL_BF16:
            scatter_custom<bfloat16_t, 2><<<blockNum, nullptr, stream>>>(
                valDev, idxDev, yDev, tiling);
            break;
        case ACL_FLOAT:
            scatter_custom<float, 4><<<blockNum, nullptr, stream>>>(
                valDev, idxDev, yDev, tiling);
            break;
        case ACL_COMPLEX64:
            // complex64 = 2×fp32 交错存储：T=float、ELEM_BYTES=8，纯字节搬运不拆实虚部
            scatter_custom<float, 8><<<blockNum, nullptr, stream>>>(
                valDev, idxDev, yDev, tiling);
            break;
        default:
            // Host 校验链保证 valueType ∈ 五 dtype 集合，default 不达；
            // 防御深度：未来 dtype 扩展遗漏校验时显式暴露，禁止静默不发射 kernel
            // （成功但无效果的假成功）。
            fprintf(stderr,
                    "aclsparseScatter: scatter_kernel_do unsupported valueType %d, kernel not launched\n",
                    static_cast<int>(valueType));
            break;
    }
}
