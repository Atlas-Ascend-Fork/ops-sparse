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
 * \file spgemm_numeric_kernel.cpp
 * \brief SpGEMM arch22 数值阶段 Kernel：按 C.rowOffsets 写出 colIndices 与 values。
 */

#include "kernel_operator.h"
#include "spgemm.h"
#include "spgemm_value.h"
#include "spgemm_merge.h"

using namespace AscendC;

namespace {

/**
 * 数值阶段 Kernel。
 *
 * @tparam T 存储类型：float / half / bfloat16_t / SpgemmComplex64
 */
template <typename T>
class SpgemmArch22NumericKernel {
public:
    static constexpr bool kIsComplex = SpgemmValueTraits<T>::kIsComplex;

    /**
     * Init 第 1 段：读入 tiling 标量字段，定位本核行区间。
     */
    __aicore__ inline void LoadTilingScalars(GM_ADDR buffer1, GM_ADDR tilingGm)
    {
        auto *td = reinterpret_cast<__gm__ SpgemmArch22TilingData *>(tilingGm);
        M_ = td->M;
        K_ = td->K;
        N_ = td->N;
        nnzA_ = td->nnzA;
        nnzB_ = td->nnzB;
        nnzCIn_ = td->nnzCIn;
        betaNonZero_ = td->betaNonZero;
        alphaRe_ = td->alphaRe;
        alphaIm_ = td->alphaIm;
        betaRe_ = td->betaRe;
        betaIm_ = td->betaIm;
        chunkWidth_ = td->chunkWidth;
        mergeCapacity_ = td->mergeCapacity;
        prodFitsCapacity_ = td->prodFitsCapacity;
        ops_.alphaRe = td->alphaRe;
        ops_.alphaIm = td->alphaIm;

        blockDim_ = GetBlockNum();
        blockId_ = GetBlockIdx();
        auto *binEdge = reinterpret_cast<__gm__ int32_t *>(buffer1 + td->binEdgeOffset);
        rowStart_ = static_cast<uint32_t>(binEdge[blockId_]);
        rowEnd_ = static_cast<uint32_t>(binEdge[blockId_ + 1]);
        auto *sb = reinterpret_cast<__gm__ int64_t *>(buffer1 + td->scratchBaseOffset);
        scratchCur_ = sb[blockId_];
        rowProductsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(buffer1 + td->rowProductsOffset), M_);
    }

    /** Init 第 2 段：绑定全部 GM 描述符。 */
    __aicore__ inline void BindGlobalBuffers(GM_ADDR aRowPtr, GM_ADDR aColIdx,
                                             GM_ADDR aValues, GM_ADDR bRowPtr,
                                             GM_ADDR bColIdx, GM_ADDR bValues,
                                             GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn,
                                             GM_ADDR cValuesIn, GM_ADDR buffer2,
                                             GM_ADDR tilingGm)
    {
        auto *td = reinterpret_cast<__gm__ SpgemmArch22TilingData *>(tilingGm);
        aRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aRowPtr), M_ + 1);
        aColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aColIdx), nnzA_);
        bRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bRowPtr), K_ + 1);
        bColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bColIdx), nnzB_);
        cRowOffGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->cRowOffsetsOffset), M_ + 1);
        cColIdxGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->cColIndicesOffset), 1);
        // values 按各类型视图访问。
        aValRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aValues), nnzA_);
        bValRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bValues), nnzB_);
        cValRawGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(buffer2 + td->cValuesOffset), 1);
        scratchColGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->scratchColOffset), 1);
        scratchValGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(buffer2 + td->scratchValOffset), 1);
        // complex64 的搬入用 int32 视图。
        scratchValIntGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->scratchValOffset), 1);
        if (betaNonZero_ != 0) {
            cRowPtrInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cRowPtrIn), M_ + 1);
            cColIdxInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cColIdxIn),
                                         nnzCIn_ > 0 ? nnzCIn_ : 1);
            cValInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cValuesIn),
                                      nnzCIn_ > 0 ? nnzCIn_ : 1);
        }
    }

    /** Init 第 3 段：申请 UB 缓冲，夹紧 chunkWidth_。 */
    __aicore__ inline void InitUbBuffers()
    {
        pipe_.InitBuffer(colQue_, 1, SPGEMM_ARCH22_COL_TILE * sizeof(int32_t));
        pipe_.InitBuffer(outColQue_, 1, mergeCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(outValQue_, 1, mergeCapacity_ * sizeof(T));

        // accBuf_ 被两条路径共用，容量取两者最大值：
        //   T1 归并：mergeCapacity_ × kAccComponents 个 float
        //   T3 分块：chunkWidth_ × (kAccComponents + 1) 个 float
        const uint32_t t1Floats = mergeCapacity_ * kAccComponents;
        const uint32_t t3Floats = chunkWidth_ * (kAccComponents + 1);
        const uint32_t accFloats = (t1Floats > t3Floats) ? t1Floats : t3Floats;
        pipe_.InitBuffer(accBuf_, accFloats * sizeof(float));
        // 夹紧 chunkWidth_ 到实际可容纳宽度。
        const uint32_t maxChunk = accFloats / (kAccComponents + 1);
        if (chunkWidth_ > maxChunk) {
            chunkWidth_ = maxChunk;
        }
    }

    /**
     * 三段顺序不可交换：GM 绑定依赖第 1 段的标量，UB 申请依赖 mergeCapacity_。
     */
    __aicore__ inline void Init(GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,
                                GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,
                                GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn, GM_ADDR cValuesIn,
                                GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tilingGm)
    {
        LoadTilingScalars(buffer1, tilingGm);
        BindGlobalBuffers(aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,
                          cRowPtrIn, cColIdxIn, cValuesIn, buffer2, tilingGm);
        InitUbBuffers();
    }

    __aicore__ inline void Process()
    {
        if (rowStart_ >= rowEnd_) {
            return;
        }
        LocalTensor<int32_t> colLocal = colQue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> outCol = outColQue_.AllocTensor<int32_t>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<T> outVal = outValQue_.AllocTensor<T>();

        // 跨行携带 CSR 行指针：上一轮的 end 即本轮的 base，避免重读。
        int32_t cEnd = cRowOffGm_.GetValue(rowStart_);
        int32_t aEnd = aRowPtrGm_.GetValue(rowStart_);
        for (uint32_t row = rowStart_; row < rowEnd_; row++) {
            const int32_t base = cEnd;
            cEnd = cRowOffGm_.GetValue(row + 1);
            const int32_t rowNnz = cEnd - base;
            const uint32_t rs = static_cast<uint32_t>(aEnd);
            aEnd = aRowPtrGm_.GetValue(row + 1);
            if (rowNnz <= 0) {
                // 空行：无输出，跳过。
                continue;
            }
            ProcessRow(row, base, rowNnz, rs, static_cast<uint32_t>(aEnd),
                       colLocal, outCol, acc, outVal);
        }
        // 尾批
        FlushCompactBatch(outCol, outVal);

        colQue_.FreeTensor(colLocal);
        outColQue_.FreeTensor(outCol);
        outValQue_.FreeTensor(outVal);
    }

private:
    static constexpr uint32_t kAccComponents = kIsComplex ? 2 : 1;

    __aicore__ inline void ProcessRow(uint32_t row, int32_t base, int32_t rowNnz,
                                      uint32_t rs, uint32_t re,
                                      const LocalTensor<int32_t> &colLocal,
                                      const LocalTensor<int32_t> &outCol,
                                      const LocalTensor<float> &acc,
                                      const LocalTensor<T> &outVal)
    {
        // rs / re 由调用方跨行携带传入。
        const uint32_t da = re - rs;

        // 分档判定与符号阶段一致。
        // prodFitsCapacity_ == 1 时可跳过逐行的 rowProducts GM 读。
        bool useChunk = (betaNonZero_ != 0) || (da > SPGEMM_ARCH22_MAX_WAYS);
        if (!useChunk && prodFitsCapacity_ == 0) {
            const int64_t prod = rowProductsGm_.GetValue(row);
            useChunk = (prod > static_cast<int64_t>(mergeCapacity_));
        }
        if (useChunk) {
            // T3 路径：完整重算本行，不使用暂存区。先 flush 已累积的批。
            FlushCompactBatch(outCol, outVal);
            ProcessRowByChunk(row, rs, re, base, rowNnz, colLocal, outCol, acc, outVal);
            return;
        }

        // ---- T1 行：从暂存区批量压实搬运到 C ----
        if (batchCount_ == 0) {
            batchDstBase_ = base;
            batchSrcBase_ = scratchCur_;
        }
        batchCount_ += static_cast<uint32_t>(rowNnz);
        scratchCur_ += rowNnz;
        if (batchCount_ + static_cast<uint32_t>(mergeCapacity_ / 2) > mergeCapacity_) {
            FlushCompactBatch(outCol, outVal);
        }
    }

    /**
     * 把已累积的一批连续 T1 行一次性从 scratch 压实搬到 C。
     */
    __aicore__ inline void FlushCompactBatch(const LocalTensor<int32_t> &outCol,
                                             const LocalTensor<T> &outVal)
    {
        if (batchCount_ == 0) {
            return;
        }
        CompactRangeFromScratch(batchDstBase_, batchSrcBase_, batchCount_, outCol, outVal);
        batchCount_ = 0;
    }

    __aicore__ inline void CompactRangeFromScratch(int32_t base, int64_t srcOff, uint32_t count,
                                                  const LocalTensor<int32_t> &outCol,
                                                  const LocalTensor<T> &outVal)
    {
        DataCopyExtParams cpCol{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padCol{false, 0, 0, 0};
        DataCopyPad(outCol, scratchColGm_[static_cast<uint64_t>(srcOff)],
                    cpCol, padCol);
        DataCopyExtParams cpVal{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        if constexpr (kIsComplex) {
            // complex64 无 DataCopyPadExtParams 特化，改用 int32 视图搬入（位级等价）。
            auto iv = outVal.template ReinterpretCast<int32_t>();
            DataCopyExtParams cpI{1, static_cast<uint32_t>(count * 2 * sizeof(int32_t)),
                                  0, 0, 0};
            DataCopyPadExtParams<int32_t> padI2{false, 0, 0, 0};
            DataCopyPad(iv, scratchValIntGm_[static_cast<uint64_t>(srcOff) * 2],
                        cpI, padI2);
        } else {
            DataCopyPadExtParams<T> padVal{false, 0, 0, 0};
            DataCopyPad(outVal, scratchValGm_[static_cast<uint64_t>(srcOff)],
                        cpVal, padVal);
        }
        // 搬入（MTE2）完成后才能搬出（MTE3）。
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(cColIdxGm_[static_cast<uint64_t>(base)], outCol, cpCol);
        DataCopyPad(cValRawGm_[static_cast<uint64_t>(base)], outVal, cpVal);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    // 注：T1 行的归并在符号阶段完成，数值阶段只做压实搬运。

    /**
     * beta·C_in 在窗口 [cbase, cbase + w) 内的累加入账。
     */
    __aicore__ inline void AccumulateBetaCIn(uint32_t crs, uint32_t cre, uint32_t cbase,
                                             uint32_t w, uint32_t markBase,
                                             const LocalTensor<float> &acc)
    {
        for (uint32_t ci = crs; ci < cre; ci++) {
            const int32_t col = cColIdxInGm_.GetValue(static_cast<uint64_t>(ci));
            if (col < static_cast<int32_t>(cbase)) {
                continue;
            }
            if (col >= static_cast<int32_t>(cbase + w)) {
                break;
            }
            float cRe = 0.0f;
            float cIm = 0.0f;
            LoadCInValue(ci, cRe, cIm);
            float pRe = 0.0f;
            float pIm = 0.0f;
            MulComplex(betaRe_, betaIm_, cRe, cIm, pRe, pIm);
            const uint32_t slot = static_cast<uint32_t>(col) - cbase;
            AddSlot(acc, slot, pRe, pIm);
            acc.SetValue(markBase + slot, 1.0f);
        }
    }

    /**
     * 单个 B 段 [bs, be) 的乘积落在窗口 [cbase, cbase + w) 内的部分入账。
     */
    __aicore__ inline void AccumulateBSegment(int32_t bs, int32_t be, float saRe, float saIm,
                                              uint32_t cbase, uint32_t w, uint32_t markBase,
                                              const LocalTensor<float> &acc)
    {
        for (int32_t bi = bs; bi < be; bi++) {
            const int32_t col = bColIdxGm_.GetValue(static_cast<uint64_t>(bi));
            if (col < static_cast<int32_t>(cbase)) {
                continue;
            }
            if (col >= static_cast<int32_t>(cbase + w)) {
                break;  // B 行内列升序，提前跳出
            }
            float bRe = 0.0f;
            float bIm = 0.0f;
            LoadBValue(static_cast<uint32_t>(bi), bRe, bIm);
            float pRe = 0.0f;
            float pIm = 0.0f;
            MulComplex(saRe, saIm, bRe, bIm, pRe, pIm);
            const uint32_t slot = static_cast<uint32_t>(col) - cbase;
            AddSlot(acc, slot, pRe, pIm);
            acc.SetValue(markBase + slot, 1.0f);
        }
    }

    /**
     * alpha·A·B 在窗口 [cbase, cbase + w) 内的累加。
     */
    __aicore__ inline void AccumulateAbWindow(uint32_t rs, uint32_t re, uint32_t cbase,
                                              uint32_t w, uint32_t markBase,
                                              const LocalTensor<int32_t> &colLocal,
                                              const LocalTensor<float> &acc)
    {
        for (uint32_t ai = rs; ai < re; ai += SPGEMM_ARCH22_COL_TILE) {
            // 瓦片搬入。
            const uint32_t len = SpgemmLoadAColTile(aColIdxGm_, ai, re, colLocal);

            for (uint32_t i = 0; i < len; i++) {
                const int32_t k = colLocal.GetValue(i);
                if (k < 0 || static_cast<uint32_t>(k) >= K_) {
                    continue;
                }
                float aRe = 0.0f;
                float aIm = 0.0f;
                LoadAValue(ai + i, aRe, aIm);
                // alpha 提前乘入 A 值。
                float saRe = 0.0f;
                float saIm = 0.0f;
                MulComplex(alphaRe_, alphaIm_, aRe, aIm, saRe, saIm);

                const int32_t bs = bRowPtrGm_.GetValue(static_cast<uint64_t>(k));
                const int32_t be = bRowPtrGm_.GetValue(static_cast<uint64_t>(k) + 1);
                AccumulateBSegment(bs, be, saRe, saIm, cbase, w, markBase, acc);
            }
        }
    }

    /**
     * 收集本窗口命中的列，按列序升序写出。
     */
    __aicore__ inline void EmitWindowHits(uint32_t cbase, uint32_t w, uint32_t markBase,
                                          int32_t base, int32_t rowNnz, int32_t &written,
                                          const LocalTensor<int32_t> &outCol,
                                          const LocalTensor<float> &acc,
                                          const LocalTensor<T> &outVal)
    {
        uint32_t pending = 0;
        for (uint32_t j = 0; j < w && written + static_cast<int32_t>(pending) < rowNnz; j++) {
            if (acc.GetValue(markBase + j) == 0.0f) {
                continue;
            }
            outCol.SetValue(pending, static_cast<int32_t>(cbase + j));
            float vRe = acc.GetValue(j * kAccComponents);
            float vIm = kIsComplex ? acc.GetValue(j * kAccComponents + 1) : 0.0f;
            StoreVal(outVal, pending, vRe, vIm);
            pending++;
            if (pending == mergeCapacity_) {
                FlushRowDirect(base + written, pending, outCol, outVal);
                written += static_cast<int32_t>(pending);
                pending = 0;
            }
        }
        if (pending > 0) {
            FlushRowDirect(base + written, pending, outCol, outVal);
            written += static_cast<int32_t>(pending);
        }
    }

    /**
     * T3 列分块稠密累加路径：把 [0, N) 切成宽度 chunkWidth 的窗口，窗口内用稠密累加器。
     * UB 占用固定，与 N 无关。beta != 0 时先并入 beta·C_in，再叠加 alpha·A·B。
     */
    __aicore__ inline void ProcessRowByChunk(uint32_t row, uint32_t rs, uint32_t re,
                                             int32_t base, int32_t rowNnz,
                                             const LocalTensor<int32_t> &colLocal,
                                             const LocalTensor<int32_t> &outCol,
                                             const LocalTensor<float> &acc,
                                             const LocalTensor<T> &outVal)
    {
        uint32_t crs = 0;
        uint32_t cre = 0;
        if (betaNonZero_ != 0) {
            crs = static_cast<uint32_t>(cRowPtrInGm_.GetValue(row));
            cre = static_cast<uint32_t>(cRowPtrInGm_.GetValue(row + 1));
        }

        // 稠密累加器与命中标记复用 accBuf_ 的前半 / 后半。
        // 布局：[0, chunkWidth×kAccComponents) 为累加值，之后 chunkWidth 个 float 作标记。
        const uint32_t width = (chunkWidth_ < N_) ? chunkWidth_ : N_;
        const uint32_t markBase = width * kAccComponents;

        int32_t written = 0;
        for (uint32_t cbase = 0; cbase < N_ && written < rowNnz; cbase += width) {
            uint32_t w = N_ - cbase;
            if (w > width) {
                w = width;
            }
            Duplicate(acc, 0.0f, static_cast<int32_t>(markBase + width));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            // beta * C_in 先入账
            AccumulateBetaCIn(crs, cre, cbase, w, markBase, acc);

            // alpha * A * B 的贡献
            AccumulateAbWindow(rs, re, cbase, w, markBase, colLocal, acc);

            EmitWindowHits(cbase, w, markBase, base, rowNnz, written, outCol, acc, outVal);
        }
    }

    // ---- 值运算：委托给 spgemm_value.h ----
    __aicore__ inline void LoadAValue(uint32_t idx, float &re, float &im)
    {
        ops_.LoadRaw(aValRawGm_, idx, re, im);
    }
    __aicore__ inline void LoadBValue(uint32_t idx, float &re, float &im)
    {
        ops_.LoadRaw(bValRawGm_, idx, re, im);
    }
    __aicore__ inline void LoadCInValue(uint32_t idx, float &re, float &im)
    {
        ops_.LoadRaw(cValInGm_, idx, re, im);
    }
    __aicore__ inline void MulComplex(float ar, float ai, float br, float bi,
                                      float &outRe, float &outIm)
    {
        ops_.MulComplex(ar, ai, br, bi, outRe, outIm);
    }
    __aicore__ inline void AddSlot(const LocalTensor<float> &acc, uint32_t slot,
                                   float re, float im)
    {
        ops_.AddSlot(acc, slot, re, im);
    }
    __aicore__ inline void StoreVal(const LocalTensor<T> &outVal, uint32_t slot,
                                    float re, float im)
    {
        ops_.StoreVal(outVal, slot, re, im);
    }

    __aicore__ inline void FlushRowDirect(int32_t base, uint32_t count,
                                          const LocalTensor<int32_t> &outCol,
                                          const LocalTensor<T> &outVal)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cpCol{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cColIdxGm_[static_cast<uint64_t>(base)], outCol, cpCol);
        DataCopyExtParams cpVal{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(cValRawGm_[static_cast<uint64_t>(base)], outVal, cpVal);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> colQue_;
    TQue<TPosition::VECOUT, 1> outColQue_;
    TQue<TPosition::VECOUT, 1> outValQue_;
    TBuf<TPosition::VECCALC> accBuf_;

    GlobalTensor<int32_t> aRowPtrGm_;
    GlobalTensor<int32_t> aColIdxGm_;
    GlobalTensor<int32_t> bRowPtrGm_;
    GlobalTensor<int32_t> bColIdxGm_;
    GlobalTensor<int32_t> cRowPtrInGm_;
    GlobalTensor<int32_t> cColIdxInGm_;
    GlobalTensor<int32_t> cRowOffGm_;
    GlobalTensor<int32_t> cColIdxGm_;
    GlobalTensor<T> aValRawGm_;
    GlobalTensor<T> bValRawGm_;
    GlobalTensor<T> cValInGm_;
    GlobalTensor<T> cValRawGm_;
    GlobalTensor<int64_t> rowProductsGm_;
    GlobalTensor<int32_t> scratchColGm_;
    GlobalTensor<T> scratchValGm_;
    GlobalTensor<int32_t> scratchValIntGm_;

    SpgemmValueOps<T> ops_;

    uint32_t M_ = 0;
    uint32_t K_ = 0;
    uint32_t N_ = 0;
    uint32_t nnzA_ = 0;
    uint32_t nnzB_ = 0;
    uint32_t nnzCIn_ = 0;
    uint32_t betaNonZero_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t blockId_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t chunkWidth_ = 32;
    uint32_t mergeCapacity_ = 64;
    // 1 表示所有行的 rowProducts <= mergeCapacity_，可跳过逐行 GM 读。
    uint32_t prodFitsCapacity_ = 0;

    // 暂存区的核内读游标，与符号阶段写游标同式递推。
    int64_t scratchCur_ = 0;
    // 批量压实：累积的元素数与批起点（C 侧 / scratch 侧）
    uint32_t batchCount_ = 0;
    int32_t batchDstBase_ = 0;
    int64_t batchSrcBase_ = 0;
    float alphaRe_ = 1.0f;
    float alphaIm_ = 0.0f;
    float betaRe_ = 0.0f;
    float betaIm_ = 0.0f;
};

}  // namespace

#define SPGEMM_DEFINE_NUMERIC_KERNEL(SUFFIX, TYPE)                                    \
    __global__ __aicore__ void spgemm_arch22_numeric_##SUFFIX(                        \
        GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,                            \
        GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,                            \
        GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn, GM_ADDR cValuesIn,                      \
        GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tilingGm)                           \
    {                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                               \
        SpgemmArch22NumericKernel<TYPE> op;                                           \
        op.Init(aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,                 \
                cRowPtrIn, cColIdxIn, cValuesIn, buffer1, buffer2, tilingGm);         \
        op.Process();                                                                 \
    }

SPGEMM_DEFINE_NUMERIC_KERNEL(fp32, float)
SPGEMM_DEFINE_NUMERIC_KERNEL(fp16, half)
SPGEMM_DEFINE_NUMERIC_KERNEL(bf16, bfloat16_t)
SPGEMM_DEFINE_NUMERIC_KERNEL(c64, SpgemmComplex64)

void spgemm_arch22_numeric_launch(
    GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,
    GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn, GM_ADDR cValuesIn,
    GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tiling,
    uint32_t dtypeId, uint32_t blockDim, void *stream)
{
#define SPGEMM_NUMERIC_LAUNCH(SUFFIX)                                                 \
    spgemm_arch22_numeric_##SUFFIX<<<blockDim, nullptr, stream>>>(                    \
        aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,                         \
        cRowPtrIn, cColIdxIn, cValuesIn, buffer1, buffer2, tiling)

    switch (dtypeId) {
        case SPGEMM_KERNEL_DTYPE_FP16:
            SPGEMM_NUMERIC_LAUNCH(fp16);
            break;
        case SPGEMM_KERNEL_DTYPE_BF16:
            SPGEMM_NUMERIC_LAUNCH(bf16);
            break;
        case SPGEMM_KERNEL_DTYPE_C64:
            SPGEMM_NUMERIC_LAUNCH(c64);
            break;
        default:
            SPGEMM_NUMERIC_LAUNCH(fp32);
            break;
    }
#undef SPGEMM_NUMERIC_LAUNCH
}
