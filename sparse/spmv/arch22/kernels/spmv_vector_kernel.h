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

#ifndef SPMV_VECTOR_KERNEL_H
#define SPMV_VECTOR_KERNEL_H

// FP32 compute path: persistent UB scratch avoids repeatedly converting and
// reallocating the same temporary storage for every CSR row.
template <typename ValT, typename OutT>
class SpmvVectorF32 {
public:
    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR columns, GM_ADDR values,
                                GM_ADDR x, GM_ADDR y, uint32_t rows, uint32_t cols,
                                float alpha, float beta)
    {
        alpha_ = alpha;
        beta_ = beta;
        const uint32_t blocks = AscendC::GetBlockNum();
        if (blocks == 0) {
            active_ = false;
            return;
        }
        const uint32_t block = AscendC::GetBlockIdx();
        rows_ = rows / blocks + (block < rows % blocks ? 1 : 0);
        const uint32_t start = block * (rows / blocks) +
            (block < rows % blocks ? block : rows % blocks);
        if (rows_ == 0) {
            active_ = false;
            return;
        }
        active_ = true;
        rowPtr_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowPtr) + start, rows_ + 1);
        startValue_ = rowPtr_(0);
        columns_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(columns) + startValue_);
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(values) + startValue_);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(x), cols);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(y) + start, rows_);
        uint32_t tile = 8;
        for (uint32_t i = 0; i < rows_; ++i) {
            int32_t n = rowPtr_(i + 1) - rowPtr_(i);
            if (n > static_cast<int32_t>(tile)) {
                tile = n;
            }
        }
        pipe_.InitBuffer(indexQueue_, 1, tile * sizeof(int32_t));
        pipe_.InitBuffer(valueQueue_, 1, tile * sizeof(float));
        shortRows_ = tile <= 8 && std::is_same_v<ValT, float> && std::is_same_v<OutT, float>;
        pipe_.InitBuffer(outputQueue_, 1, shortRows_ ?
            SPMV_ATOMIC_BATCH * SPMV_ATOMIC_ALIGNMENT : sizeof(float));
        pipe_.InitBuffer(xBuffer_, tile * sizeof(float));
        pipe_.InitBuffer(reduceBuffer_, tile * sizeof(float));
        pipe_.InitBuffer(scratchBuffer_, tile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        if constexpr (std::is_same_v<ValT, float> && std::is_same_v<OutT, float>) {
            if (shortRows_) {
                ProcessShortRows();
                return;
            }
        }
        for (uint32_t row = 0; row < rows_; ++row)
            ProcessRow(row);
    }

private:
    __aicore__ inline void ProcessShortRows()
    {
        auto batch = outputQueue_.AllocTensor<float>();
        SpmvScalarRows(pipe_, batch, rowPtr_, columns_, values_, x_, y_,
                       rows_, startValue_, alpha_, beta_);
        outputQueue_.FreeTensor(batch);
    }

    __aicore__ inline void CopyInRow(uint32_t row, int32_t count,
                                     AscendC::LocalTensor<int32_t> indices,
                                     AscendC::LocalTensor<ValT> values,
                                     AscendC::LocalTensor<OutT> output)
    {
        if (count > 0) {
            const uint32_t offset = rowPtr_(row) - startValue_;
            AscendC::DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyExtParams vc{1, static_cast<uint32_t>(count * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> ip{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> vp{false, 0, 0, 0};
            AscendC::DataCopyPad(indices, columns_[offset], ic, ip);
            AscendC::DataCopyPad(values, values_[offset], vc, vp);
        }
        if (beta_ != 0.0f) {
            AscendC::DataCopyExtParams yc{1, static_cast<uint32_t>(sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> yp{false, 0, 0, 0};
            AscendC::DataCopyPad(output, y_[row], yc, yp);
        }
        indexQueue_.EnQue(indices);
        valueQueue_.EnQue(values);
    }

    __aicore__ inline void ComputeProduct(int32_t count,
                                          AscendC::LocalTensor<int32_t> indices,
                                          AscendC::LocalTensor<ValT> values,
                                          AscendC::LocalTensor<float> x,
                                          AscendC::LocalTensor<float> reduce,
                                          AscendC::LocalTensor<float> scratch)
    {
        if (count == 0) {
            AscendC::Duplicate(scratch, 0.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        SpmvSyncEvent<AscendC::HardEvent::MTE2_S>(pipe_);
        if constexpr (std::is_same_v<ValT, float>) {
            for (int32_t i = 0; i < count; ++i)
                x.SetValue(i, x_(indices(i)));
            SpmvSyncEvent<AscendC::HardEvent::S_V>(pipe_);
            AscendC::Mul(x, x, values, count);
        } else {
            auto rawX = scratchBuffer_.Get<ValT>();
            for (int32_t i = 0; i < count; ++i)
                rawX.SetValue(i, x_(indices(i)));
            SpmvSyncEvent<AscendC::HardEvent::S_V>(pipe_);
            AscendC::Cast(x, rawX, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(reduce, values, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(x, x, reduce, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(scratch, x, reduce, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (alpha_ != 1.0f) {
            AscendC::Muls(scratch, scratch, alpha_, 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void AccumulateOutput(AscendC::LocalTensor<OutT> output,
                                            AscendC::LocalTensor<float> x,
                                            AscendC::LocalTensor<float> scratch)
    {
        if constexpr (std::is_same_v<OutT, float>) {
            if (beta_ == 0.0f) {
                AscendC::Adds(output, scratch, 0.0f, 1);
            } else {
                if (beta_ != 1.0f) {
                    AscendC::Muls(output, output, beta_, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(output, output, scratch, 1);
            }
        } else {
            if (beta_ != 0.0f) {
                AscendC::Cast(x, output, AscendC::RoundMode::CAST_NONE, 1);
                AscendC::PipeBarrier<PIPE_V>();
                if (beta_ != 1.0f) {
                    AscendC::Muls(x, x, beta_, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(scratch, scratch, x, 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(output, scratch, AscendC::RoundMode::CAST_ROUND, 1);
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const int32_t count = rowPtr_(row + 1) - rowPtr_(row);
        auto indices = indexQueue_.AllocTensor<int32_t>();
        auto values = valueQueue_.AllocTensor<ValT>();
        auto output = outputQueue_.AllocTensor<OutT>();
        auto x = xBuffer_.Get<float>();
        auto reduce = reduceBuffer_.Get<float>();
        auto scratch = scratchBuffer_.Get<float>();
        CopyInRow(row, count, indices, values, output);
        indices = indexQueue_.DeQue<int32_t>();
        values = valueQueue_.DeQue<ValT>();
        ComputeProduct(count, indices, values, x, reduce, scratch);
        AccumulateOutput(output, x, scratch);
        AscendC::PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<OutT>();
        AscendC::DataCopyExtParams oc{1, static_cast<uint32_t>(sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(y_[row], output, oc);
        if (beta_ != 0.0f) {
            SpmvSyncEvent<AscendC::HardEvent::MTE3_MTE2>(pipe_);
        }
        outputQueue_.FreeTensor(output);
        valueQueue_.FreeTensor(values);
        indexQueue_.FreeTensor(indices);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indexQueue_, valueQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuffer_, reduceBuffer_, scratchBuffer_;
    AscendC::GlobalTensor<int32_t> rowPtr_, columns_;
    AscendC::GlobalTensor<ValT> values_, x_;
    AscendC::GlobalTensor<OutT> y_;
    bool shortRows_ = false;
    bool active_ = false;
    uint32_t rows_ = 0;
    uint32_t startValue_ = 0;
    float alpha_, beta_;
};
#endif
