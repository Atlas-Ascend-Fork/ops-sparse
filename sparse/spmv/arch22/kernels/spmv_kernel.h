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
#pragma once

#include "kernel_operator.h"
#include "../spmv_tiling_data.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPMV_ATOMIC_BATCH = 128;
constexpr uint32_t SPMV_ATOMIC_ALIGNMENT = 32;

// These helpers make the producer/consumer relationship explicit for temporary
// UB buffers that cross hardware pipelines or are reused by another pipeline.
// Reserve an ID for the complete Set/Wait pair, then release it.
// Batching keeps this management outside the per-nonzero DMA loop.
template <AscendC::HardEvent Event>
__aicore__ inline void SpmvSyncEvent(AscendC::TPipe &pipe)
{
    event_t eventId = static_cast<event_t>(pipe.AllocEventID<Event>());
    AscendC::SetFlag<Event>(eventId);
    AscendC::WaitFlag<Event>(eventId);
    pipe.ReleaseEventID<Event>(eventId);
}

// Same-type short rows use Scalar for immutable CSR/x/y reads and batch
// independent row outputs. Each DMA source starts at a distinct 32-byte slot.
template <typename T>
__aicore__ inline void SpmvScalarRows(
    AscendC::TPipe &pipe, AscendC::LocalTensor<T> batch,
    AscendC::GlobalTensor<int32_t> &rowPtr, AscendC::GlobalTensor<int32_t> &columns,
    AscendC::GlobalTensor<T> &values, AscendC::GlobalTensor<T> &x,
    AscendC::GlobalTensor<T> &y, uint32_t rows, int32_t startValue, T alpha, T beta)
{
    constexpr uint32_t stride = SPMV_ATOMIC_ALIGNMENT / sizeof(T);
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    for (uint32_t first = 0; first < rows; first += SPMV_ATOMIC_BATCH) {
        uint32_t count = rows - first;
        if (count > SPMV_ATOMIC_BATCH) {
            count = SPMV_ATOMIC_BATCH;
        }
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t row = first + i;
            T sum = 0;
            int32_t begin = rowPtr(row) - startValue;
            int32_t end = rowPtr(row + 1) - startValue;
            for (int32_t j = begin; j < end; ++j) {
                if constexpr (std::is_same_v<T, int32_t>) {
                    // Unsigned arithmetic defines INT32 wrap without C++ signed overflow.
                    sum = static_cast<int32_t>(static_cast<uint32_t>(sum) +
                        static_cast<uint32_t>(values(j)) * static_cast<uint32_t>(x(columns(j))));
                } else {
                    sum += values(j) * x(columns(j));
                }
            }
            T result;
            if constexpr (std::is_same_v<T, int32_t>) {
                uint32_t v = static_cast<uint32_t>(sum) * static_cast<uint32_t>(alpha);
                if (beta != 0) {
                    v += static_cast<uint32_t>(beta) * static_cast<uint32_t>(y(row));
                }
                result = static_cast<int32_t>(v);
            } else {
                result = alpha * sum;
                if (beta != 0) {
                    result += beta * y(row);
                }
            }
            batch.SetValue(i * stride, result);
        }
        SpmvSyncEvent<AscendC::HardEvent::S_MTE3>(pipe);
        for (uint32_t i = 0; i < count; ++i)
            AscendC::DataCopyPad(y[first + i], batch[i * stride], cp);
        SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
    }
}

/**
 * @brief CSR 格式稀疏矩阵-稠密向量乘法（SpMV）AscendC 算子
 *
 * 采用行并行策略：每个 AI Core 处理若干整行，
 * 每行内通过 CopyIn → Compute → CopyOut 三级流水完成计算。
 *
 * @tparam T 数据类型（支持 float / int32_t）
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernel {
public:
    __aicore__ inline SpmvKernel(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
    __aicore__ inline void CopyOut(int32_t currentRow);
    __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx, inQueueVals;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX, workQueueReduce, floatQueueReduce;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta) {
    this->totalRowsNum = totalRowsNum;
    this->totalColsNum = totalColsNum;
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0) {
        this->blockRowNum = 0;
        return;
    }

    // 均分行：前 tempRowNum 个核各多拿一行
    uint32_t tempRowNum = totalRowsNum % blockNum;
    if (AscendC::GetBlockIdx() < tempRowNum) {
        this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum() + 1;
        this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
    }
    else {
        this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum();
        this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
    }

    if (this->blockRowNum == 0) {
        return;
    }

    // 每核只切出自己负责的 rowPtr 段 + vals/colIdx 段
    csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow, this->blockRowNum + 1);

    this->startValIdx = csrRowPtrGm(0);
    this->endValIdx = csrRowPtrGm(this->blockRowNum);
    if (endValIdx >= startValIdx) {
        this->blockLength = endValIdx - startValIdx;
    } else {
        this->blockLength = 0;
    }

    csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx, this->blockLength);
    csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx, this->blockLength);
    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalColsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec + this->startRow, this->blockRowNum);

    // 计算当前核处理行中的最大行长（tile 大小依据）
    this->tileLength = 0;
    for (int i = 0; i < this->blockRowNum; i++) {
        // Guard against non-monotonic rowPtr: negative (int32) length must not
        // wrap to a huge uint32_t and blow up InitBuffer (issues #147/#148).
        int32_t rowStart = csrRowPtrGm(i);
        int32_t rowEnd = csrRowPtrGm(i + 1);
        if (rowEnd < rowStart) {
            continue;
        }
        uint32_t rowLength = static_cast<uint32_t>(rowEnd - rowStart);
        if (rowLength > this->tileLength) {
            this->tileLength = rowLength;
        }
    }
    if (this->tileLength == 0) {
        this->tileLength = 8; // 全空行也给最小对齐量，避免 InitBuffer(0)
    }

    // 通过共享 UB 的精度进行计算优化
    pipe.InitBuffer(inQueueColIdx, BUFFER_NUM, this->tileLength * sizeof(int32_t));
    pipe.InitBuffer(inQueueVals, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueX, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(floatQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength <= 8 ?
        SPMV_ATOMIC_BATCH * SPMV_ATOMIC_ALIGNMENT : sizeof(CompT));
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyIn(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.AllocTensor<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
    if (validNum > 0) {
        AscendC::DataCopyExtParams copyParamsColIdx{1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

        uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
        AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx, padParamsColIdx);

        if constexpr (std::is_same_v<ValT, CompT>) {
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
        }
        else {
            // Cast 矢量指令（新 CANN 版本验证）
            AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
            SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);
            AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
            SpmvSyncEvent<AscendC::HardEvent::V_MTE2>(pipe);
            workQueueX.FreeTensor(valsTmp);
        }
    }
    inQueueColIdx.EnQue(colIdxLocal);
    inQueueVals.EnQue(valsLocal);

    // Read y from GM for beta*y computation in Compute
    AscendC::LocalTensor<CompT> yLocal = outQueueY.AllocTensor<CompT>();
    if (this->beta == static_cast<CompT>(0)) {
        AscendC::Duplicate(yLocal, static_cast<CompT>(0), 1);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        if constexpr (std::is_same_v<OutT, CompT>) {
            AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(CompT), 0, 0, 0};
            AscendC::DataCopyPadExtParams<CompT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yLocal, yVecGm[currentRow], cp, padParams);
            SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);
        }
        else {
            AscendC::LocalTensor<OutT> yTmp = workQueueX.AllocTensor<OutT>();
            AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(OutT), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yTmp, yVecGm[currentRow], cp, padParams);
            SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);
            AscendC::Cast<CompT, OutT>(yLocal, yTmp,
                                       AscendC::RoundMode::CAST_NONE, 1);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
            SpmvSyncEvent<AscendC::HardEvent::V_MTE2>(pipe);
            workQueueX.FreeTensor(yTmp);
        }
    }
    outQueueY.EnQue(yLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyOut(int32_t currentRow) {
    AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();
    if constexpr (std::is_same_v<CompT, OutT>) {
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        // VECOUT EnQue/DeQue already orders vector production before MTE3.
        AscendC::DataCopyPad(yVecGm[currentRow], yLocal, copyParams);
    }
    else {
        AscendC::LocalTensor<OutT> yOut = workQueueX.AllocTensor<OutT>();
        AscendC::Cast<OutT, CompT>(yOut, yLocal,
                                   AscendC::RoundMode::CAST_ROUND, 1);
        AscendC::PipeBarrier<PIPE_V>();
        SpmvSyncEvent<AscendC::HardEvent::V_MTE3>(pipe);
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[currentRow], yOut, copyParams);
        SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
        workQueueX.FreeTensor(yOut);
    }
    // The next row reloads yLocal through MTE2 after this MTE3 store.
    SpmvSyncEvent<AscendC::HardEvent::MTE3_MTE2>(pipe);
    outQueueY.FreeTensor(yLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Compute(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();
    AscendC::LocalTensor<CompT> xLocal = workQueueX.AllocTensor<CompT>();
    // Delay the scalar dependency until consumption; CopyIn can first issue
    // the independent values/y loads instead of stalling after the index DMA.
    if (validNum > 0) {
        SpmvSyncEvent<AscendC::HardEvent::MTE2_S>(pipe);
    }
    AscendC::LocalTensor<float> sharedTmpBuffer;
    AscendC::LocalTensor<float> floatTmpBuffer;

    if (validNum > 0) {
        if constexpr (std::is_same_v<ValT, CompT>) {
            for (int i = 0; i < validNum; i++) {
                ValT v = xVecGm(colIdxLocal(i));
                xLocal.SetValue(i, v);
            }
            SpmvSyncEvent<AscendC::HardEvent::S_V>(pipe);
        }
        else {
            // ValT≠CompT: Cast 转换，从 floatQueueReduce 分配 xTmp 避免 workQueueX 冲突
            AscendC::LocalTensor<ValT> xTmp = floatQueueReduce.AllocTensor<ValT>();
            for (int i = 0; i < validNum; i++) {
                ValT v = xVecGm(colIdxLocal(i));
                xTmp.SetValue(i, v);
            }
            SpmvSyncEvent<AscendC::HardEvent::S_V>(pipe);
            AscendC::Cast<CompT, ValT>(xLocal, xTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
            floatQueueReduce.FreeTensor(xTmp);
        }

        // 逐元素乘：xLocal *= valsLocal
        AscendC::Mul(xLocal, xLocal, valsLocal, validNum);
        AscendC::PipeBarrier<PIPE_V>();

        sharedTmpBuffer = workQueueReduce.AllocTensor<float>();
        floatTmpBuffer = floatQueueReduce.AllocTensor<float>();

        // ReduceSum 在 float 做，其余计算以 CompT 进行
        if constexpr (std::is_same_v<CompT, float>) {
            // float 路径：xLocal 就是 float，不需要 Cast，直接 ReduceSum
            AscendC::ReduceSum<float>(floatTmpBuffer, xLocal,
                                      sharedTmpBuffer, validNum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();

            if (this->alpha != static_cast<CompT>(1)) {
                AscendC::Muls(floatTmpBuffer, floatTmpBuffer, this->alpha, 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (this->beta == static_cast<CompT>(0)) {
                AscendC::Adds(yLocal, floatTmpBuffer, static_cast<CompT>(0), 1);
            } else {
                if (this->beta != static_cast<CompT>(1)) {
                    AscendC::Muls(yLocal, yLocal, this->beta, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(yLocal, yLocal, floatTmpBuffer, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            outQueueY.EnQue<CompT>(yLocal);
        }
        else {
            // 非 float 路径：Cast→float→ReduceSum→Cast 回 CompT
            AscendC::Cast<float, CompT>(floatTmpBuffer, xLocal,
                                        AscendC::RoundMode::CAST_ROUND, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(floatTmpBuffer, floatTmpBuffer,
                                      sharedTmpBuffer, validNum);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();

            // Cast sum from float back to CompT（复用 xLocal）
            AscendC::Cast<CompT, float>(xLocal, floatTmpBuffer,
                                        AscendC::RoundMode::CAST_ROUND, 1);
            AscendC::PipeBarrier<PIPE_V>();

            // z = alpha * sum + beta * y   (all CompT)
            AscendC::Muls(xLocal, xLocal, this->alpha, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yLocal, yLocal, this->beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(yLocal, yLocal, xLocal, 1);
            AscendC::PipeBarrier<PIPE_V>();

            outQueueY.EnQue<CompT>(yLocal);
        }

        workQueueReduce.FreeTensor(sharedTmpBuffer);
        floatQueueReduce.FreeTensor(floatTmpBuffer);
    }
    else {
        // 空行：z = beta * y  (CompT)
        AscendC::LocalTensor<CompT> yLocal = outQueueY.DeQue<CompT>();
        AscendC::Muls(yLocal, yLocal, this->beta, 1);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY.EnQue<CompT>(yLocal);
    }

    inQueueColIdx.FreeTensor(colIdxLocal);
    inQueueVals.FreeTensor(valsLocal);
    workQueueX.FreeTensor(xLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Process() {
    if (this->blockRowNum == 0) {
        return;
    }

    if constexpr (std::is_same_v<CompT, ValT> && std::is_same_v<CompT, OutT>) {
        if (tileLength <= 8) {
            auto batch = outQueueY.AllocTensor<CompT>();
            SpmvScalarRows(pipe, batch, csrRowPtrGm, csrColIndGm, csrValGm,
                           xVecGm, yVecGm, blockRowNum, startValIdx, alpha, beta);
            outQueueY.FreeTensor(batch);
            return;
        }
    }

    for (int32_t m = 0; m < this->blockRowNum; m++) {
        int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
        CopyIn(m, validNum);
        Compute(m, validNum);
        CopyOut(m);
    }
}

/**
 * @brief CSR 格式稀疏矩阵转置-稠密向量乘法（SpMV Transpose）AscendC 算子
 *
 * 采用与 SpmvKernel 一致的行并行策略：每个 AI Core 处理若干整行，
 * 但结果 y 不再是一行一个标量，而是一行中每个非零元对 y 的某个列产生贡献。
 * 多行对同一 y[j] 的累加通过 atomic add 避免核间踩踏。
 *
 * @tparam T 数据类型（支持 float / int32_t）
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernelTrans {
public:
    __aicore__ inline SpmvKernelTrans(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessScalarSmallRows();
    __aicore__ inline void FlushScalarBatch(AscendC::LocalTensor<CompT> &batch, uint32_t count);
    __aicore__ inline void ScaleBeta();
    __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
    __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);
    __aicore__ inline void CopyOut(int32_t currentRow, int32_t validNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx, inQueueVals;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX, workQueueReduce, floatQueueReduce;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;

    // beta 缩放阶段：当前核负责的 y 段（读）
    uint32_t yBlockStart;
    uint32_t yBlockCount;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta) {
    this->totalRowsNum = totalRowsNum;
    this->totalColsNum = totalColsNum;
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum > 0) {
        // ===== 行分布（与 SpmvKernel 一致） =====
        uint32_t tempRowNum = totalRowsNum % blockNum;
        if (AscendC::GetBlockIdx() < tempRowNum) {
            this->blockRowNum = this->totalRowsNum / blockNum + 1;
            this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
        }
        else {
            this->blockRowNum = this->totalRowsNum / blockNum;
            this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
        }

        // ===== y 列分布（beta 缩放阶段：均分 totalColsNum 个 y 元素到各核） =====
        uint32_t tempColNum = totalColsNum % blockNum;
        if (AscendC::GetBlockIdx() < tempColNum) {
            this->yBlockCount = this->totalColsNum / blockNum + 1;
            this->yBlockStart = this->yBlockCount * AscendC::GetBlockIdx();
        }
        else {
            this->yBlockCount = this->totalColsNum / blockNum;
            this->yBlockStart = this->yBlockCount * AscendC::GetBlockIdx() + tempColNum;
        }
    }
    else {
        this->blockRowNum = 0;
        this->yBlockCount = 0;
    }

    // xVecGm/yVecGm 对所有核均需初始化（ScaleBeta 在所有核上运行）
    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalRowsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec, this->totalColsNum);

    if (this->blockRowNum > 0) {
        csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow, this->blockRowNum + 1);

        this->startValIdx = csrRowPtrGm(0);
        this->endValIdx = csrRowPtrGm(this->blockRowNum);
        if (endValIdx >= startValIdx) {
            this->blockLength = endValIdx - startValIdx;
        } else {
            this->blockLength = 0;
        }

        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx, this->blockLength);
        csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx, this->blockLength);

        this->tileLength = 0;
        for (int i = 0; i < this->blockRowNum; i++) {
            // Guard against non-monotonic rowPtr (issues #147/#148).
            int32_t rowStart = csrRowPtrGm(i);
            int32_t rowEnd = csrRowPtrGm(i + 1);
            if (rowEnd < rowStart) {
                continue;
            }
            uint32_t rowLength = static_cast<uint32_t>(rowEnd - rowStart);
            if (rowLength > this->tileLength) {
                this->tileLength = rowLength;
            }
        }
        if (this->tileLength == 0) {
            this->tileLength = 8;
        }
    }
    else {
        this->tileLength = 8;
    }

    // UB 缓冲区初始化（所有核均需初始化 pipe）
    pipe.InitBuffer(inQueueColIdx, BUFFER_NUM, this->tileLength * sizeof(int32_t));
    pipe.InitBuffer(inQueueVals, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueX, BUFFER_NUM, this->tileLength * sizeof(CompT));
    pipe.InitBuffer(workQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(floatQueueReduce, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, SPMV_ATOMIC_BATCH * SPMV_ATOMIC_ALIGNMENT);
}

/**
 * @brief beta 缩放阶段：每个核将自己负责的 y 段乘以 beta
 *
 * 由于多核间 y 段不重叠，无需原子操作。
 * 当 beta == 0 时 host 端已做 y 全零初始化，本函数直接返回。
 * 当 beta == 1 时无需缩放，直接返回。
 */
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ScaleBeta() {
    if (this->yBlockCount == 0) {
        return;
    }

    // beta == 1：无需缩放
    if (this->beta == static_cast<CompT>(1)) {
        return;
    }

    // 以 tileLength 为块大小逐块处理 y[ yBlockStart .. yBlockStart+yBlockCount )
    for (uint32_t offset = 0; offset < this->yBlockCount; offset += this->tileLength) {
        uint32_t chunkSize = this->tileLength;
        if (offset + chunkSize > this->yBlockCount) {
            chunkSize = this->yBlockCount - offset;
        }

        if constexpr (std::is_same_v<CompT, OutT>) {
            AscendC::LocalTensor<CompT> yChunkLocal = workQueueX.AllocTensor<CompT>();

            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(CompT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<CompT> padParams{false, 0, 0, 0};
            if (this->beta == static_cast<CompT>(0)) {
                AscendC::Duplicate(yChunkLocal, static_cast<CompT>(0), chunkSize);
            } else {
                AscendC::DataCopyPad(yChunkLocal,
                                     yVecGm[this->yBlockStart + offset],
                                     cp, padParams);
                SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);
                AscendC::Muls(yChunkLocal, yChunkLocal, this->beta, chunkSize);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(CompT)), 0, 0, 0};
            SpmvSyncEvent<AscendC::HardEvent::V_MTE3>(pipe);
            AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkLocal, cp2);
            // The next chunk reloads this workQueueX buffer through MTE2.
            SpmvSyncEvent<AscendC::HardEvent::MTE3_MTE2>(pipe);

            workQueueX.FreeTensor(yChunkLocal);
        }
        else {
            // OutT ≠ CompT (e.g. half output with float compute)
            // Read y as OutT → cast to CompT → *beta → cast back → write y
            AscendC::LocalTensor<OutT> yChunkOut = workQueueX.AllocTensor<OutT>();
            AscendC::LocalTensor<CompT> yChunkComp =
                floatQueueReduce.AllocTensor<CompT>();

            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunkOut,
                                 yVecGm[this->yBlockStart + offset],
                                 cp, padParams);
            SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);

            AscendC::Cast<CompT, OutT>(yChunkComp, yChunkOut,
                                       AscendC::RoundMode::CAST_NONE, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(yChunkComp, yChunkComp, this->beta, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<OutT, CompT>(yChunkOut, yChunkComp,
                                       AscendC::RoundMode::CAST_ROUND, chunkSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            SpmvSyncEvent<AscendC::HardEvent::V_MTE3>(pipe);
            AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkOut, cp2);
            // The next chunk reloads this workQueueX buffer through MTE2.
            SpmvSyncEvent<AscendC::HardEvent::MTE3_MTE2>(pipe);

            floatQueueReduce.FreeTensor(yChunkComp);
            workQueueX.FreeTensor(yChunkOut);
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyIn(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.AllocTensor<int32_t>();
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
    if (validNum > 0) {
        AscendC::DataCopyExtParams copyParamsColIdx{1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

        uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
        AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx, padParamsColIdx);

        if constexpr (std::is_same_v<ValT, CompT>) {
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
        }
        else {
            AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
            AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
            AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
            SpmvSyncEvent<AscendC::HardEvent::MTE2_V>(pipe);
            AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                       AscendC::RoundMode::CAST_NONE, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
            SpmvSyncEvent<AscendC::HardEvent::V_MTE2>(pipe);
            workQueueX.FreeTensor(valsTmp);
        }
    }
    inQueueColIdx.EnQue(colIdxLocal);
    inQueueVals.EnQue(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Compute(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();
    AscendC::LocalTensor<CompT> contribLocal = workQueueX.AllocTensor<CompT>();

    if (validNum > 0) {
        CompT xVal;
        if constexpr (std::is_same_v<ValT, CompT>) {
            xVal = xVecGm(this->startRow + currentRow);
        }
        else {
            AscendC::LocalTensor<ValT> xTmp = outQueueY.AllocTensor<ValT>();
            AscendC::LocalTensor<CompT> cTmp = floatQueueReduce.AllocTensor<CompT>();
            xTmp.SetValue(0, xVecGm(this->startRow + currentRow));
            SpmvSyncEvent<AscendC::HardEvent::S_V>(pipe);
            AscendC::Cast<CompT, ValT>(cTmp, xTmp,
                                       AscendC::RoundMode::CAST_NONE, 1);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
            xVal = cTmp.GetValue(0);
            outQueueY.FreeTensor(xTmp);
            floatQueueReduce.FreeTensor(cTmp);
        }

        AscendC::Muls(contribLocal, valsLocal, xVal, validNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(contribLocal, contribLocal,
                      this->alpha, validNum);
        AscendC::PipeBarrier<PIPE_V>();
        // CopyOut reads contribLocal through scalar GetValue.
        SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);
    }
    workQueueX.EnQue(contribLocal);
    inQueueVals.FreeTensor(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyOut(int32_t currentRow, int32_t validNum) {
    AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
    AscendC::LocalTensor<CompT> contribLocal = workQueueX.DeQue<CompT>();

    if (validNum > 0) {
        SpmvSyncEvent<AscendC::HardEvent::MTE2_S>(pipe);
        if constexpr (std::is_same_v<CompT, OutT>) {
            AscendC::LocalTensor<CompT> atomicElem = outQueueY.AllocTensor<CompT>();
            // Each UB source starts on a 32-byte boundary. Stage a batch
            // before DMA so the source is never overwritten while MTE3 reads it.
            constexpr uint32_t stride = SPMV_ATOMIC_ALIGNMENT / sizeof(CompT);
            for (int base = 0; base < validNum; base += SPMV_ATOMIC_BATCH) {
                int n = validNum - base < SPMV_ATOMIC_BATCH ? validNum - base : SPMV_ATOMIC_BATCH;
                for (int j = 0; j < n; ++j)
                    atomicElem.SetValue(j * stride, contribLocal(base + j));
                SpmvSyncEvent<AscendC::HardEvent::S_MTE3>(pipe);
                AscendC::SetAtomicAdd<CompT>();
                AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(CompT), 0, 0, 0};
                for (int j = 0; j < n; ++j)
                    AscendC::DataCopyPad(yVecGm[colIdxLocal(base + j)], atomicElem[j * stride], cp);
                AscendC::SetAtomicNone();
                SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
            }
            outQueueY.FreeTensor(atomicElem);
        }
        else {
            // CompT≠OutT: cast contrib→OutT，用 OutT 做原子累加直接写入 yVecGm
            AscendC::LocalTensor<OutT> oCast = floatQueueReduce.AllocTensor<OutT>();

            AscendC::Cast<OutT, CompT>(oCast, contribLocal,
                                       AscendC::RoundMode::CAST_ROUND, validNum);
            AscendC::PipeBarrier<PIPE_V>();
            SpmvSyncEvent<AscendC::HardEvent::V_S>(pipe);

            AscendC::LocalTensor<OutT> atomicElem = outQueueY.AllocTensor<OutT>();
            // Each UB source starts on a 32-byte boundary. Stage a batch
            // before DMA so the source is never overwritten while MTE3 reads it.
            constexpr uint32_t stride = SPMV_ATOMIC_ALIGNMENT / sizeof(OutT);
            for (int base = 0; base < validNum; base += SPMV_ATOMIC_BATCH) {
                int n = validNum - base < SPMV_ATOMIC_BATCH ? validNum - base : SPMV_ATOMIC_BATCH;
                for (int j = 0; j < n; ++j)
                    atomicElem.SetValue(j * stride, oCast(base + j));
                SpmvSyncEvent<AscendC::HardEvent::S_MTE3>(pipe);
                AscendC::SetAtomicAdd<OutT>();
                AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(OutT), 0, 0, 0};
                for (int j = 0; j < n; ++j)
                    AscendC::DataCopyPad(yVecGm[colIdxLocal(base + j)], atomicElem[j * stride], cp);
                AscendC::SetAtomicNone();
                SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
            }
            outQueueY.FreeTensor(atomicElem);
            floatQueueReduce.FreeTensor(oCast);
        }
    }

    inQueueColIdx.FreeTensor(colIdxLocal);
    workQueueX.FreeTensor(contribLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::FlushScalarBatch(
    AscendC::LocalTensor<CompT> &batch, uint32_t count)
{
    auto targets = batch.template ReinterpretCast<int32_t>();
    constexpr uint32_t stride = SPMV_ATOMIC_ALIGNMENT / sizeof(CompT);
    SpmvSyncEvent<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::SetAtomicAdd<CompT>();
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(CompT)), 0, 0, 0};
    for (uint32_t i = 0; i < count; ++i)
        AscendC::DataCopyPad(yVecGm[targets(i * stride + 1)], batch[i * stride], cp);
    AscendC::SetAtomicNone();
    SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ProcessScalarSmallRows()
{
    // Short CSR rows do not amortize per-row vector setup. Immutable CSR/x
    // inputs can be read by Scalar; only the staged output crosses to MTE3.
    auto batch = outQueueY.AllocTensor<CompT>();
    auto targets = batch.template ReinterpretCast<int32_t>();
    constexpr uint32_t stride = SPMV_ATOMIC_ALIGNMENT / sizeof(CompT);
    uint32_t pending = 0;
    for (uint32_t row = 0; row < blockRowNum; ++row) {
        int32_t begin = csrRowPtrGm(row) - startValIdx;
        int32_t end = csrRowPtrGm(row + 1) - startValIdx;
        if (end <= begin) {
            continue;
        }
        ValT rawX = xVecGm(startRow + row);
        CompT xv;
        if constexpr (std::is_same_v<ValT, bfloat16_t>) {
            xv = AscendC::Cast(rawX);
        } else {
            xv = static_cast<CompT>(rawX);
        }
        for (int32_t j = begin; j < end; ++j) {
            ValT rawValue = csrValGm(j);
            CompT value;
            if constexpr (std::is_same_v<ValT, bfloat16_t>) {
                value = AscendC::Cast(rawValue);
            } else {
                value = static_cast<CompT>(rawValue);
            }
            if constexpr (std::is_same_v<CompT, int32_t>) {
                batch.SetValue(pending * stride, static_cast<int32_t>(
                    static_cast<uint32_t>(value) * static_cast<uint32_t>(xv) * static_cast<uint32_t>(alpha)));
            } else {
                batch.SetValue(pending * stride, (value * xv) * alpha);
            }
            targets.SetValue(pending * stride + 1, csrColIndGm(j));
            if (++pending == SPMV_ATOMIC_BATCH) {
                FlushScalarBatch(batch, pending);
                pending = 0;
            }
        }
    }
    if (pending > 0) {
        FlushScalarBatch(batch, pending);
    }
    outQueueY.FreeTensor(batch);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Process() {
    // 阶段 1：beta 缩放（多核分工无重叠，无需原子）
    ScaleBeta();
    // SyncAll is issued by scalar; wait until every local beta write has left MTE3.
    SpmvSyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
    AscendC::SyncAll();

    // Batch short rows across row boundaries rather than waiting per row.
    if constexpr (std::is_same_v<CompT, OutT>) {
        if (tileLength <= 8) {
            ProcessScalarSmallRows();
            AscendC::SyncAll();
            return;
        }
    }
    // 阶段 2：逐行处理，每行贡献原子累加到 y
    for (int32_t m = 0; m < this->blockRowNum; m++) {
        int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
        if (validNum <= 0) {
            continue;
        }
        CopyIn(m, validNum);
        Compute(m, validNum);
        CopyOut(m, validNum);
    }
    AscendC::SyncAll();
}

// -------------------------------------------------------------------
// 具体 Kernel 入口 + Host 端启动封装
// -------------------------------------------------------------------

#include "spmv_vector_kernel.h"

// ---- per-type __global__ kernel 定义（各自 TU 展开一个实例） ----
#define DEFINE_SPMV_KERNEL(CompT, ValT, OutT)                                                           \
    __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT(                                   \
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,                                           \
        GM_ADDR xVec, GM_ADDR yVec,                                                                      \
        uint32_t totalRowsNum, uint32_t totalColNum,                                                     \
        CompT alpha, CompT beta, bool trans)                                                            \
    {                                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                 \
        if (trans)                                                                                      \
        {                                                                                               \
            SpmvKernelTrans<CompT, ValT, OutT> op;                                                      \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum, alpha, beta);  \
            op.Process();                                                                               \
        }                                                                                               \
        else                                                                                            \
        {                                                                                               \
            if constexpr (std::is_same_v<CompT, float>) {                                               \
                SpmvVectorF32<ValT, OutT> op;                                                           \
                op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum, alpha, beta); \
                op.Process();                                                                           \
            } else {                                                                                   \
                SpmvKernel<CompT, ValT, OutT> op;                                                        \
                op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum, alpha, beta); \
                op.Process();                                                                           \
            }                                                                                          \
        }                                                                                               \
    }

// ---- extern 声明（供 spmv_kernel_do 引用各 TU 中的 __global__ kernel） ----
#define SPMV_EXTERN(CompT, ValT, OutT) \
    extern __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT( \
        GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, CompT, CompT, bool);
SPMV_EXTERN(float, float, float)
SPMV_EXTERN(float, half, half)
SPMV_EXTERN(float, half, float)
SPMV_EXTERN(float, bfloat16_t, bfloat16_t)
SPMV_EXTERN(float, bfloat16_t, float)
SPMV_EXTERN(float, int32_t, float)
SPMV_EXTERN(int32_t, int32_t, int32_t)
#undef SPMV_EXTERN
