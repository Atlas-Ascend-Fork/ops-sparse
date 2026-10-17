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
 * \file spgemm_symbolic_kernel.cpp
 * \brief SpGEMM arch22 符号阶段 Kernel：确定 C 的结构（rowOffsets 与 nnz(C)）。
 *
 * 与 dtype 完全无关——只读 A/B 的 rowOffsets/colIndices，不读 values，
 * 故 fp16/bf16/fp32/complex64 四种类型共用本 Kernel。
 *
 * 两级前缀和在同一个 Kernel 内完成，避免额外的 kernel 下发：
 *   1) 各核串行扫描自己的行区间，得到局部前缀和与局部总和 coreSum[core]
 *   2) SyncAll() 后，每核读取前序各核的局部总和作为基址偏移，写出全局 C.rowOffsets
 *
 * 行内算法分档（与数值阶段判定式完全一致）：
 *   da ≤ SPGEMM_ARCH22_MAX_WAYS → T1 k 路有序归并
 *   否则                        → T3 列分块位图
 */

#include "kernel_operator.h"
#include "spgemm.h"
#include "spgemm_value.h"
#include "spgemm_merge.h"


using namespace AscendC;

namespace {

// 向量化发射暂存的容量（元素数），取自 spgemm_merge.h 的定义。
constexpr uint32_t kVecEmitCap = kSpgemmVecEmitCap;
// B 行长统计的每核槽位跨距（int32 个数，= 64B 一整条 cacheline，防伪共享）。
// 必须与 spgemm_count_kernel.cpp 的 kStatStride 一致。
constexpr uint32_t kBStatStride = 16;

template <typename T, bool COALESCE_B = false>
class SpgemmArch22SymbolicKernel {
public:
    static constexpr bool kIsComplex = SpgemmValueTraits<T>::kIsComplex;

    /** Init 第 1 段：读 tiling 标量，定位本核行区间与融合暂存区起始槽位。 */
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
        mergeCapacity_ = td->mergeCapacity;
        ops_.alphaRe = td->alphaRe;
        ops_.alphaIm = td->alphaIm;

        blockDim_ = GetBlockNum();
        blockId_ = GetBlockIdx();

        auto *binEdge = reinterpret_cast<__gm__ int32_t *>(buffer1 + td->binEdgeOffset);
        rowStart_ = static_cast<uint32_t>(binEdge[blockId_]);
        rowEnd_ = static_cast<uint32_t>(binEdge[blockId_ + 1]);
        auto *sb = reinterpret_cast<__gm__ int64_t *>(buffer1 + td->scratchBaseOffset);
        scratchBase_ = sb[blockId_];
        scratchLimit_ = sb[blockId_ + 1];
        scratchCur_ = scratchBase_;
    }

    /**
     * Init 第 2 段：绑定 GM 描述符。
     * LoadBUniformLen() 必须在 bStatGm_ 绑定之后调用。
     */
    __aicore__ inline void BindGlobalBuffers(GM_ADDR aRowPtr, GM_ADDR aColIdx,
                                             GM_ADDR aValues, GM_ADDR bRowPtr,
                                             GM_ADDR bColIdx, GM_ADDR bValues,
                                             GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn,
                                             GM_ADDR buffer1, GM_ADDR buffer2,
                                             GM_ADDR tilingGm)
    {
        auto *td = reinterpret_cast<__gm__ SpgemmArch22TilingData *>(tilingGm);
        aRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aRowPtr), M_ + 1);
        aColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aColIdx), nnzA_);
        bRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bRowPtr), K_ + 1);
        bColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bColIdx), nnzB_);
        // B 每行等差公差描述符（count Kernel 产出，本趟只读）。
        bApStrideGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bApStrideOffset),
            (K_ > 0) ? K_ : 1);
        // B 每行首列索引（count Kernel 产出，本趟只读）。
        bHeadGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bHeadOffset),
            (K_ > 0) ? K_ : 1);
        // B 行长统计（count Kernel 发布），本趟只读、只归约一次。
        bStatGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bStatOffset),
            blockDim_ * kBStatStride);
        LoadBUniformLen();
        rowNnzGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->rowNnzOffset), M_);
        rowProductsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(buffer1 + td->rowProductsOffset), M_);
        coreSumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->coreSumOffset), blockDim_ + 1);
        // 软件 SyncAll 的屏障区（与 count 共用同一段，host 在本趟 launch 前清零）。
        syncGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->syncOffset), blockDim_ * 8);
        cRowOffGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->cRowOffsetsOffset), M_ + 1);
        // Variant B：融合路径需要读 A/B 的值，并写出 (col, val) 暂存。
        aValGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(aValues), nnzA_);
        bValGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bValues), nnzB_);
        scratchColGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer2 + td->scratchColOffset), 1);
        scratchValGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(buffer2 + td->scratchValOffset), 1);
        if (betaNonZero_ != 0) {
            cRowPtrInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cRowPtrIn), M_ + 1);
            cColIdxInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cColIdxIn),
                                         nnzCIn_ > 0 ? nnzCIn_ : 1);
        }
    }

    /** Init 第 3 段：申请 UB 队列与融合路径的输出暂存。 */
    __aicore__ inline void InitUbBuffers()
    {
        pipe_.InitBuffer(colQue_, 1, SPGEMM_ARCH22_COL_TILE * sizeof(int32_t));
        pipe_.InitBuffer(outQue_, 1, SPGEMM_ARCH22_ROW_TILE * sizeof(int32_t));
        pipe_.InitBuffer(markBuf_, SPGEMM_ARCH22_SYM_CHUNK * sizeof(int32_t));
        pipe_.InitBuffer(fusedColQue_, 1, mergeCapacity_ * sizeof(int32_t));
        pipe_.InitBuffer(fusedValQue_, 1, mergeCapacity_ * sizeof(T));
        // 窄类型把 fp32 累加值先存这里，flush 时一条向量 Cast 批量转换成目标类型。
        if constexpr (SpgemmValueOps<T>::kDeferCast) {
            pipe_.InitBuffer(wideValBuf_, mergeCapacity_ * sizeof(float));
            wideVal_ = wideValBuf_.template Get<float>();
        }
    }

    /**
     * Init 第 4 段：在 markBuf_ 上切出向量发射路径的各视图，并预置 V→S 令牌。
     * 必须在 InitUbBuffers() 之后（markBuf_ 已申请）。
     *
     * 向量发射暂存全部借用 markBuf_，不新增 UB 分配。markBuf_ 只被 T3 列分块使用，
     * T3 与向量发射同属单行内的临时用途、绝不交叠。
     */
    __aicore__ inline void InitVecEmitContext()
    {
        static_assert(20 * kVecEmitCap <= SPGEMM_ARCH22_SYM_CHUNK,
                      "vec emit views must fit in markBuf_");
        vecCtx_.off = markBuf_.template Get<int32_t>();
        vecCtx_.offU = markBuf_.template Get<uint32_t>();
        vecCtx_.tile = markBuf_.template Get<float>()[kVecEmitCap];
        vecCtx_.aTile = markBuf_.template Get<float>()[2 * kVecEmitCap];
        if constexpr (SpgemmValueOps<T>::kDeferCast) {  // NOLINT
            vecCtx_.narrow = markBuf_.template Get<T>()[(3 * kVecEmitCap) *
                                                        (sizeof(float) / sizeof(T))];
        }
        // 层号表与暂存，起址 4/5/6·cap，均 32B 对齐且不与 narrow 重叠。
        vecCtx_.layerIdx = markBuf_.template Get<int32_t>()[4 * kVecEmitCap];
        vecCtx_.layerBytes = markBuf_.template Get<int32_t>()[5 * kVecEmitCap];
        vecCtx_.layerCol = markBuf_.template Get<int32_t>()[6 * kVecEmitCap];
        vecCtx_.cap = kVecEmitCap;
        vecCtx_.enabled = true;
        BuildVecLayerTables();
        // 预置向量发射路径的 V→S 标志。
        // EVENT_ID6 上恒有且仅有一个 V→S 令牌在飞：
        //   置：这里预置一个（否则第一个发射行的 Wait 会挂死）；
        //   转：每个发射行「块首消一个、块尾置一个」；
        //   消：Process 里 CountRows 返回后消掉最后一个。
        SetFlag<HardEvent::V_S>(EVENT_ID6);
    }

    /** 四段顺序不可交换：后三段都消费第 1 段读出的 M_/K_/nnz*_ 与 blockDim_。 */
    __aicore__ inline void Init(GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,
                                GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,
                                GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn,
                                GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tilingGm)
    {
        LoadTilingScalars(buffer1, tilingGm);
        BindGlobalBuffers(aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,
                          cRowPtrIn, cColIdxIn, buffer1, buffer2, tilingGm);
        InitUbBuffers();
        InitVecEmitContext();
    }

    __aicore__ inline void Process()
    {
        CountRows();
        // 消掉跨行流转的 V→S 令牌。放在 SyncAll 之前。
        WaitFlag<HardEvent::V_S>(EVENT_ID6);
        // 软件 SyncAll（不触发 FFTS）。
        LocalTensor<int32_t> syncLocal = markBuf_.template Get<int32_t>();
        SyncAll(syncGm_, syncLocal, static_cast<int32_t>(blockDim_));
        WriteRowOffsets();
    }

private:
    /** 实数路径的发射表（da=8 层号表 + 段号表 + vec7 表）。 */
    __aicore__ inline void BuildVecRealTables()
    {
        constexpr uint32_t da = SpgemmVecEmitCtx<T>::kVecDa;
        for (uint32_t j = 0; j < kVecEmitCap; j++) {
            const int32_t t = static_cast<int32_t>(j / da);
            vecCtx_.layerIdx.SetValue(j, t);
            vecCtx_.layerBytes.SetValue(j, t * static_cast<int32_t>(sizeof(float)));
        }
        // da=8 段号表 iPat8[j] = j % 8，铺前 8 个后倍增摊出。
        {
            const LocalTensor<int32_t> iPat8 =
                markBuf_.template Get<int32_t>()[kSpgemmVec8ModIdx * kVecEmitCap];
            for (uint32_t j = 0; j < da; j++) {
                iPat8.SetValue(j, static_cast<int32_t>(j));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID7);
            WaitFlag<HardEvent::S_V>(EVENT_ID7);
            uint32_t len = da;
            while (len < kVecEmitCap) {
                uint32_t n = kVecEmitCap - len;
                if (n > len) { n = len; }
                Adds(iPat8[len], iPat8, 0, static_cast<int32_t>(n));
                len += n;
            }
        }
        BuildVec7Tables();
    }

    /** complex64 路径的发射表（交错表 + 三张行不变索引表）。 */
    __aicore__ inline void BuildVecComplexTables()
    {
        constexpr uint32_t da = SpgemmVecEmitCtx<T>::kVecDa;
        // 交错写出索引表：ilv[2j] = 实部第 j 个，ilv[2j+1] = 虚部第 j 个。
        const LocalTensor<uint32_t> ilv =
            markBuf_.template Get<uint32_t>()[18 * kVecEmitCap];
        for (uint32_t j = 0; j < kVecEmitCap; j++) {
            ilv.SetValue(2 * j, j * static_cast<uint32_t>(sizeof(float)));
            ilv.SetValue(2 * j + 1,
                         (kVecEmitCap + j) * static_cast<uint32_t>(sizeof(float)));
        }
        // c64 发射块的三张行不变索引表：iPat/layI/layB8/bcast。
        // 铺前 da 个后倍增摊出，bcast 由 iPat 一条 Muls 导出。
        const LocalTensor<int32_t> pool = markBuf_.template Get<int32_t>();
        const LocalTensor<int32_t> iPat = pool[3 * kVecEmitCap];
        const LocalTensor<int32_t> layI = pool[6 * kVecEmitCap];
        const LocalTensor<int32_t> layB8 = pool[7 * kVecEmitCap];
        const LocalTensor<int32_t> bcast = pool[4 * kVecEmitCap];
        for (uint32_t i = 0; i < da; i++) {
            iPat.SetValue(i, static_cast<int32_t>(i));
            layI.SetValue(i, 0);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        {
            uint32_t len = da;
            while (len < kVecEmitCap) {
                uint32_t n = kVecEmitCap - len;
                if (n > len) { n = len; }
                Adds(iPat[len], iPat, 0, static_cast<int32_t>(n));
                Adds(layI[len], layI, static_cast<int32_t>(len / da),
                     static_cast<int32_t>(n));
                len += n;
            }
        }
        constexpr int32_t kCplxB = 2 * static_cast<int32_t>(sizeof(float));
        Muls(layB8, layI, kCplxB, static_cast<int32_t>(kVecEmitCap));
        Muls(bcast, iPat, kCplxB, static_cast<int32_t>(kVecEmitCap));
    }

    /**
     * 按类型编译期择一建表（if constexpr），末尾 S_V 同步是两臂共同的收口。
     */
    __aicore__ inline void BuildVecLayerTables()
    {
        if constexpr (!SpgemmValueOps<T>::kIsComplex) {
            BuildVecRealTables();
        } else {
            BuildVecComplexTables();
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
    }

    /**
     * da=7 向量发射所需的四张行不变表。
     *   mod7Bytes[j]   = 4·(j % 7)
     *   layer7Idx[j]   = j / 7
     *   layer7Bytes[j] = 4·(j / 7)
     *   shift[p·64+j]  = 4·(j − p)（j ≥ p；j < p 填 0）
     *
     * 铺 lcm(7,8)=56 个 + 倍增，把标量写压到最少。
     */
    __aicore__ inline void BuildVec7Tables()
    {
        static_assert(kVecEmitCap >= 256u,
                      "vec7 shift table assumes cap >= 256 (它占 2·cap 的分区)");
        static_assert(kSpgemmVec7Base * kVecEmitCap + 24u <= SPGEMM_ARCH22_SYM_CHUNK,
                      "vec7 views must fit in markBuf_");
        static_assert((kSpgemmVec8ModIdx + 1u) * kVecEmitCap <= SPGEMM_ARCH22_SYM_CHUNK,
                      "iPat8 view must fit in markBuf_");
        // A 值窄类型暂存槽容量断言。
        static_assert((kSpgemmVec8ARaw + 1u) * kVecEmitCap <= SPGEMM_ARCH22_SYM_CHUNK,
                      "A-raw staging block must fit in markBuf_");
        constexpr uint32_t kSeg = 7u;
        constexpr uint32_t kPat = 56u;   // lcm(7, 8)
        constexpr int32_t kB = static_cast<int32_t>(sizeof(float));
        const LocalTensor<int32_t> pool = markBuf_.template Get<int32_t>();
        const LocalTensor<int32_t> mod7 = pool[kSpgemmVec7Mod * kVecEmitCap];
        const LocalTensor<int32_t> lay7 = pool[kSpgemmVec7Layer * kVecEmitCap];
        const LocalTensor<int32_t> lay7B = pool[kSpgemmVec7LayerB * kVecEmitCap];
        const LocalTensor<int32_t> shift = pool[kSpgemmVec7Shift * kVecEmitCap];
        // mod7Idx[j] = j % 7（纯下标，不带 ×4）。
        const LocalTensor<int32_t> mod7I = pool[kSpgemmVec7ModIdx * kVecEmitCap];
        for (uint32_t j = 0; j < kPat; j++) {
            mod7I.SetValue(j, static_cast<int32_t>(j % kSeg));
            lay7.SetValue(j, static_cast<int32_t>(j / kSeg));
        }
        for (uint32_t j = 0; j < 8u; j++) {
            shift.SetValue(j, static_cast<int32_t>(j) * kB);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        uint32_t len = kPat;
        while (len < kVecEmitCap) {
            uint32_t n = kVecEmitCap - len;
            if (n > len) { n = len; }
            Adds(mod7I[len], mod7I, 0, static_cast<int32_t>(n));
            Adds(lay7[len], lay7, static_cast<int32_t>(len / kSeg),
                 static_cast<int32_t>(n));
            len += n;
        }
        Muls(mod7, mod7I, kB, static_cast<int32_t>(kVecEmitCap));
        Muls(lay7B, lay7, kB, static_cast<int32_t>(kVecEmitCap));
        BuildVec7ShiftTail(shift, kB);
    }

    /**
     * vec7 shift 表收尾：倍增铺满 [0,64)，按周期 64 复制出 p=1..7 的七份并各减 p·4，
     * 填零前 p 道避免产生缓冲外地址。
     */
    __aicore__ inline void BuildVec7ShiftTail(const LocalTensor<int32_t> &shift, int32_t kB)
    {
        uint32_t rl = 8u;
        while (rl < 64u) {
            Adds(shift[rl], shift, static_cast<int32_t>(rl) * kB,
                 static_cast<int32_t>(rl));
            rl += rl;
        }
        for (uint32_t p = 1u; p < 8u; p++) {
            Adds(shift[p * 64u], shift, -static_cast<int32_t>(p) * kB, 64);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (uint32_t p = 1u; p < 8u; p++) {
            for (uint32_t j = 0; j < p; j++) {
                shift.SetValue(p * 64u + j, 0);
            }
        }
    }

    /** 写出本核局部总和到 coreSum[blockId_+1]（coreSum[0] 恒为 0）。 */
    __aicore__ inline void WriteCoreSum(const LocalTensor<int32_t> &outLocal,
                                        int64_t localTotal)
    {
        outLocal.SetValue(0, static_cast<int32_t>(localTotal));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(coreSumGm_[blockId_ + 1], outLocal, cp);
        if (blockId_ == 0) {
            outLocal.SetValue(0, 0);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(coreSumGm_[0], outLocal, cp);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // 第 1 步：数出本核负责区间内每行的 nnz，并累出本核局部总和
    __aicore__ inline void CountRows()
    {
        LocalTensor<int32_t> outLocal = outQue_.AllocTensor<int32_t>();
        int64_t localTotal = 0;

        if (rowStart_ < rowEnd_) {
            LocalTensor<int32_t> colLocal = colQue_.AllocTensor<int32_t>();
            LocalTensor<int32_t> fusedCol = fusedColQue_.template AllocTensor<int32_t>();
            LocalTensor<T> fusedVal = fusedValQue_.template AllocTensor<T>();
            // 每批设一次批基址（行循环之外）。
            if constexpr (!SpgemmValueOps<T>::kIsComplex) {
                vecCtx_.baseCol = fusedCol;
                if constexpr (SpgemmValueOps<T>::kDeferCast) {
                    vecCtx_.baseDst = wideVal_;
                } else {
                    vecCtx_.baseDst = fusedVal;
                }
            }
            uint32_t pending = 0;
            uint32_t flushBase = rowStart_;

            for (uint32_t row = rowStart_; row < rowEnd_; row++) {
                const int32_t nnz = CountOneRow(row, colLocal, fusedCol, fusedVal);
                // 写出核内独占前缀和（本行之前所有行的 nnz 之和）。
                outLocal.SetValue(pending, static_cast<int32_t>(localTotal));
                localTotal += nnz;
                pending++;
                if (pending == SPGEMM_ARCH22_ROW_TILE) {
                    FlushRowNnz(outLocal, flushBase, pending);
                    flushBase = row + 1;
                    pending = 0;
                }
            }
            if (pending > 0) {
                FlushRowNnz(outLocal, flushBase, pending);
            }
            FlushBatch(fusedCol, fusedVal);
            fusedValQue_.FreeTensor(fusedVal);
            fusedColQue_.FreeTensor(fusedCol);
            colQue_.FreeTensor(colLocal);
        }

        WriteCoreSum(outLocal, localTotal);
        outQue_.FreeTensor(outLocal);
    }

    /**
     * 单行 nnz 计数。行内算法与数值阶段判定式一致：
     *   da ≤ MAX_WAYS → T1 归并；否则 → T3 列分块位图。
     */
    __aicore__ inline int32_t CountOneRow(uint32_t row, const LocalTensor<int32_t> &colLocal,
                                          const LocalTensor<int32_t> &fusedCol,
                                          const LocalTensor<T> &fusedVal)
    {
        const uint32_t rs = static_cast<uint32_t>(aRowPtrGm_.GetValue(row));
        const uint32_t re = static_cast<uint32_t>(aRowPtrGm_.GetValue(row + 1));
        const uint32_t da = re - rs;
        // prod 由 uniform-B 证书解析计算（band_.len > 0 时），否则读 GM。
        const int32_t bandLen = band_.len;
        const int32_t daS = static_cast<int32_t>(da);
        const int64_t prod = (bandLen > 0)
                                 ? ((daS > 0) ? static_cast<int64_t>(daS) *
                                                    static_cast<int64_t>(bandLen)
                                              : static_cast<int64_t>(0))
                                 : rowProductsGm_.GetValue(row);


        // 分档判定式与数值阶段完全一致（Variant B 核心不变式）。
        const bool useChunk = (betaNonZero_ != 0) ||
                              (da > SPGEMM_ARCH22_MAX_WAYS) ||
                              (prod > static_cast<int64_t>(mergeCapacity_));
        if (useChunk) {
            FlushBatch(fusedCol, fusedVal);
            return CountRowByChunk(row, rs, re, colLocal);
        }
        if (da == 0) { return 0; }

        SpgemmSegments seg;
        if (!SpgemmLoadSegmentsGmWithVal(aColIdxGm_, rs, da, bRowPtrGm_, K_, bApStrideGm_,
                                         bHeadGm_, aValGm_, ops_, band_, seg)) {
            FlushBatch(fusedCol, fusedVal);
            return CountRowByChunk(row, rs, re, colLocal);
        }


        // Variant B：单趟融合，归并同时算出数值写入暂存区。
        // 剩余容量不足时先 flush。
        if (batchCount_ + static_cast<uint32_t>(prod) > mergeCapacity_) {
            FlushBatch(fusedCol, fusedVal);
        }

        // aBase：段 s 与 A 表项 aBase+s 一一对应时为非负，否则 −1。
        const int32_t aBase = (seg.count == da) ? static_cast<int32_t>(rs) : -1;
        const int32_t nnz = SpgemmMergeFusedRow<COALESCE_B>(bColIdxGm_, bValGm_, ops_, seg,
                                                fusedCol[batchCount_], fusedVal[batchCount_],
                                                wideVal_[batchCount_],
                                                mergeCapacity_ - batchCount_, vecCtx_,
                                                batchCount_, aValGm_, aBase);
        if (nnz < 0) {
            // 不可达兜底：重新计数并推进暂存游标（保持两趟递推同步）。
            FlushBatch(fusedCol, fusedVal);
            SpgemmSegments seg2;
            if (!SpgemmLoadSegmentsGm(aColIdxGm_, rs, da, bRowPtrGm_, K_, bApStrideGm_,
                                      bHeadGm_, seg2)) {
                return CountRowByChunk(row, rs, re, colLocal);
            }
            const int32_t n2 = SpgemmMergeCountRow<COALESCE_B>(bColIdxGm_, seg2);
            scratchCur_ += n2;
            return n2;
        }
        if (nnz > 0) { batchCount_ += static_cast<uint32_t>(nnz); }
        return nnz;
    }

    /**
     * 跨核归约 B 的行长统计，得到「全局等长」的长度 L（不等长则 0）。
     * 语义与 count Kernel 的 LoadBStat 一致。
     */
    __aicore__ inline void LoadBUniformLen()
    {
        int32_t gMin = 0x7FFFFFFF;
        int32_t gMax = -0x7FFFFFFF;
        int32_t gBandBad = 0x7FFFFFFF;
        for (uint32_t c = 0; c < blockDim_; c++) {
            const uint32_t slot = c * kBStatStride;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                    DcciDst::CACHELINE_OUT>(bStatGm_[slot]);
            const int32_t lo = bStatGm_.GetValue(slot);
            const int32_t hi = bStatGm_.GetValue(slot + 1);
            if (lo < gMin) {
                gMin = lo;
            }
            if (hi > gMax) {
                gMax = hi;
            }
            const int32_t bb = bStatGm_.GetValue(slot + 2);
            if (bb < gBandBad) {
                gBandBad = bb;
            }
        }
        const int32_t uniformLen = ((K_ > 0) && (gMin == gMax) && (gMin > 0)) ? gMin : 0;
        // 带要求 bRowPtr[k] == k·L，以全局等长为前提。
        band_.len = uniformLen;
        band_.stride = 0;
        band_.head = 0;
        band_.prefix = 0;
        if ((uniformLen > 0) && (gBandBad > 0)) {
            uint32_t p = static_cast<uint32_t>(gBandBad);
            if (p > K_) {
                p = K_;
            }
            band_.prefix = p;
            band_.stride = bApStrideGm_.GetValue(0);
            band_.head = bHeadGm_.GetValue(0);
        }
    }

    __aicore__ inline void FlushBatch(const LocalTensor<int32_t> &fusedCol,
                                      const LocalTensor<T> &fusedVal)
    {
        if (batchCount_ == 0) {
            return;
        }
        FlushFusedRow(fusedCol, fusedVal, batchCount_);
        batchCount_ = 0;
    }

    __aicore__ inline void FlushFusedRow(const LocalTensor<int32_t> &fusedCol,
                                         const LocalTensor<T> &fusedVal, uint32_t count)
    {
        if (scratchCur_ + count > scratchLimit_) {
            // 越界防御：必须推进游标以保持两趟递推同步。
            scratchCur_ += count;
            return;
        }
        // 窄类型在此处批量 Cast fp32 → 目标类型（round-half-to-even）。
        if constexpr (SpgemmValueOps<T>::kDeferCast) {
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Cast(fusedVal, wideVal_, RoundMode::CAST_RINT, count);
        }
        // 等标量和向量流水都排空后 MTE3 才能搬。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cpCol{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(scratchColGm_[static_cast<uint64_t>(scratchCur_)], fusedCol, cpCol);
        DataCopyExtParams cpVal{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(scratchValGm_[static_cast<uint64_t>(scratchCur_)], fusedVal, cpVal);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        scratchCur_ += count;
    }

    /** 在 [base, base + width) 窗口内标记 A 行经 B 展开后命中的列。 */
    __aicore__ inline void MarkAbHitsInWindow(uint32_t rs, uint32_t re, uint32_t base,
                                              uint32_t width,
                                              const LocalTensor<int32_t> &mark,
                                              const LocalTensor<int32_t> &colLocal)
    {
        for (uint32_t ai = rs; ai < re; ai += SPGEMM_ARCH22_COL_TILE) {
            const uint32_t len = SpgemmLoadAColTile(aColIdxGm_, ai, re, colLocal);

            for (uint32_t i = 0; i < len; i++) {
                const int32_t k = colLocal.GetValue(i);
                if (k < 0 || static_cast<uint32_t>(k) >= K_) {
                    continue;
                }
                const int32_t bs = bRowPtrGm_.GetValue(static_cast<uint64_t>(k));
                const int32_t be = bRowPtrGm_.GetValue(static_cast<uint64_t>(k) + 1);
                MarkBSegmentInWindow(bs, be, base, width, mark);
            }
        }
    }

    /** 把 B 段 [bs, be) 落在窗口内的列打上标记。升序超右界即跳出。 */
    __aicore__ inline void MarkBSegmentInWindow(int32_t bs, int32_t be, uint32_t base,
                                                uint32_t width,
                                                const LocalTensor<int32_t> &mark)
    {
        for (int32_t bi = bs; bi < be; bi++) {
            const int32_t col = bColIdxGm_.GetValue(static_cast<uint64_t>(bi));
            if (col < static_cast<int32_t>(base)) {
                continue;
            }
            if (col >= static_cast<int32_t>(base + width)) {
                break;
            }
            mark.SetValue(static_cast<uint32_t>(col) - base, 1);
        }
    }

    /**
     * T3 列分块位图计数：把 [0, N) 切成宽度 SYM_CHUNK 的窗口，每窗口标记命中列。
     * UB 占用固定为 SYM_CHUNK × 4B，与 N 无关。
     */
    __aicore__ inline int32_t CountRowByChunk(uint32_t row, uint32_t rs, uint32_t re,
                                              const LocalTensor<int32_t> &colLocal)
    {
        LocalTensor<int32_t> mark = markBuf_.Get<int32_t>();
        int32_t total = 0;
        const uint32_t chunk = SPGEMM_ARCH22_SYM_CHUNK;

        uint32_t crs = 0;
        uint32_t cre = 0;
        if (betaNonZero_ != 0) {
            crs = static_cast<uint32_t>(cRowPtrInGm_.GetValue(row));
            cre = static_cast<uint32_t>(cRowPtrInGm_.GetValue(row + 1));
        }

        for (uint32_t base = 0; base < N_; base += chunk) {
            uint32_t width = N_ - base;
            if (width > chunk) {
                width = chunk;
            }
            Duplicate(mark, 0, static_cast<int32_t>(chunk));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            MarkAbHitsInWindow(rs, re, base, width, mark, colLocal);
            // 并入 C 原有结构（beta != 0）
            for (uint32_t ci = crs; ci < cre; ci++) {
                const int32_t col = cColIdxInGm_.GetValue(static_cast<uint64_t>(ci));
                if (col < static_cast<int32_t>(base)) {
                    continue;
                }
                if (col >= static_cast<int32_t>(base + width)) {
                    break;
                }
                mark.SetValue(static_cast<uint32_t>(col) - base, 1);
            }

            for (uint32_t j = 0; j < width; j++) {
                total += mark.GetValue(j);
            }
        }
        // T3 位图用满 markBuf_，覆盖了向量发射的表，在此重建。
        BuildVecLayerTables();
        return total;
    }

    __aicore__ inline void FlushRowNnz(const LocalTensor<int32_t> &outLocal,
                                       uint32_t base, uint32_t count)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(rowNnzGm_[base], outLocal, cp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // 第 2 步：核间偏移 + 写出全局 C.rowOffsets
    __aicore__ inline void WriteRowOffsets()
    {
        int32_t base = 0;
        for (uint32_t c = 0; c <= blockId_; c++) {
            base += coreSumGm_.GetValue(c);
        }

        LocalTensor<int32_t> outLocal = outQue_.AllocTensor<int32_t>();
        if (rowStart_ < rowEnd_) {
            // 整块平移：cRowOff = 核内前缀和 + base。
            LocalTensor<int32_t> preLocal = colQue_.AllocTensor<int32_t>();
            DataCopyPadExtParams<int32_t> padIn{false, 0, 0, 0};
            for (uint32_t row = rowStart_; row < rowEnd_;
                 row += SPGEMM_ARCH22_ROW_TILE) {
                uint32_t cnt = rowEnd_ - row;
                if (cnt > SPGEMM_ARCH22_ROW_TILE) { cnt = SPGEMM_ARCH22_ROW_TILE; }
                DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * sizeof(int32_t)),
                                     0, 0, 0};
                DataCopyPad(preLocal, rowNnzGm_[row], cp, padIn);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Adds(outLocal, preLocal, base, cnt);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(cRowOffGm_[row], outLocal, cp);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
            colQue_.FreeTensor(preLocal);
            if (rowEnd_ == M_) {
                outLocal.SetValue(0, base + coreSumGm_.GetValue(blockId_ + 1));
                FlushRowOff(outLocal, M_, 1);
            }
        } else if (M_ == 0 && blockId_ == 0) {
            outLocal.SetValue(0, 0);
            FlushRowOff(outLocal, 0, 1);
        }
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void FlushRowOff(const LocalTensor<int32_t> &outLocal,
                                       uint32_t base, uint32_t count)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cRowOffGm_[base], outLocal, cp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> colQue_;
    TQue<TPosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> markBuf_;
    TQue<TPosition::VECOUT, 1> fusedColQue_;
    TQue<TPosition::VECOUT, 1> fusedValQue_;

    GlobalTensor<int32_t> aRowPtrGm_;
    GlobalTensor<int32_t> aColIdxGm_;
    GlobalTensor<int32_t> bRowPtrGm_;
    GlobalTensor<int32_t> bColIdxGm_;
    GlobalTensor<int32_t> bApStrideGm_;
    GlobalTensor<int32_t> bHeadGm_;
    GlobalTensor<int32_t> bStatGm_;
    // B 的等长与带状前缀描述。
    SpgemmBandInfo band_;
    GlobalTensor<int32_t> cRowPtrInGm_;
    GlobalTensor<int32_t> cColIdxInGm_;
    GlobalTensor<int32_t> rowNnzGm_;
    GlobalTensor<int64_t> rowProductsGm_;
    GlobalTensor<int32_t> coreSumGm_;
    // 软件 SyncAll 的 GM 屏障区。
    GlobalTensor<int32_t> syncGm_;
    GlobalTensor<int32_t> cRowOffGm_;
    GlobalTensor<T> aValGm_;
    GlobalTensor<T> bValGm_;
    GlobalTensor<int32_t> scratchColGm_;
    GlobalTensor<T> scratchValGm_;

    SpgemmValueOps<T> ops_;

    uint32_t M_ = 0;
    uint32_t K_ = 0;
    uint32_t N_ = 0;
    uint32_t nnzA_ = 0;
    uint32_t nnzB_ = 0;
    uint32_t nnzCIn_ = 0;
    uint32_t betaNonZero_ = 0;
    uint32_t mergeCapacity_ = 0;
    SpgemmVecEmitCtx<T> vecCtx_;
    TBuf<TPosition::VECCALC> wideValBuf_;
    LocalTensor<float> wideVal_;
    uint32_t blockDim_ = 1;
    uint32_t blockId_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowEnd_ = 0;
    int64_t scratchBase_ = 0;
    int64_t scratchCur_ = 0;
    int64_t scratchLimit_ = 0;
    uint32_t batchCount_ = 0;
};

}  // namespace

#define SPGEMM_DEFINE_SYMBOLIC_KERNEL(NAME, TYPE, COALESCE_B)                         \
    __global__ __aicore__ void NAME(                                                   \
        GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,                            \
        GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,                            \
        GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn,                                         \
        GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tilingGm)                           \
    {                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                               \
        SpgemmArch22SymbolicKernel<TYPE, COALESCE_B> op;                              \
        op.Init(aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,                 \
                cRowPtrIn, cColIdxIn, buffer1, buffer2, tilingGm);                    \
        op.Process();                                                                 \
    }

SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_fp32, float, false)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_fp16, half, false)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_bf16, bfloat16_t, false)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_c64, SpgemmComplex64, false)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_fp32_duplicates, float, true)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_fp16_duplicates, half, true)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_bf16_duplicates, bfloat16_t, true)
SPGEMM_DEFINE_SYMBOLIC_KERNEL(spgemm_arch22_symbolic_c64_duplicates, SpgemmComplex64, true)

void spgemm_arch22_symbolic_launch(
    GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR aValues,
    GM_ADDR bRowPtr, GM_ADDR bColIdx, GM_ADDR bValues,
    GM_ADDR cRowPtrIn, GM_ADDR cColIdxIn,
    GM_ADDR buffer1, GM_ADDR buffer2, GM_ADDR tiling,
    uint32_t dtypeId, bool coalesceB, uint32_t blockDim, void *stream)
{
#define SPGEMM_SYMBOLIC_LAUNCH(SUFFIX)                                                \
    spgemm_arch22_symbolic_##SUFFIX<<<blockDim, nullptr, stream>>>(                   \
        aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,                         \
        cRowPtrIn, cColIdxIn, buffer1, buffer2, tiling)
#define SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH(SUFFIX)                                     \
    spgemm_arch22_symbolic_##SUFFIX##_duplicates<<<blockDim, nullptr, stream>>>(      \
        aRowPtr, aColIdx, aValues, bRowPtr, bColIdx, bValues,                         \
        cRowPtrIn, cColIdxIn, buffer1, buffer2, tiling)

    if (coalesceB) {
        switch (dtypeId) {
            case SPGEMM_KERNEL_DTYPE_FP16: SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH(fp16); break;
            case SPGEMM_KERNEL_DTYPE_BF16: SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH(bf16); break;
            case SPGEMM_KERNEL_DTYPE_C64: SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH(c64); break;
            default: SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH(fp32); break;
        }
    } else switch (dtypeId) {
        case SPGEMM_KERNEL_DTYPE_FP16:
            SPGEMM_SYMBOLIC_LAUNCH(fp16);
            break;
        case SPGEMM_KERNEL_DTYPE_BF16:
            SPGEMM_SYMBOLIC_LAUNCH(bf16);
            break;
        case SPGEMM_KERNEL_DTYPE_C64:
            SPGEMM_SYMBOLIC_LAUNCH(c64);
            break;
        default:
            SPGEMM_SYMBOLIC_LAUNCH(fp32);
            break;
    }
#undef SPGEMM_SYMBOLIC_LAUNCH
#undef SPGEMM_SYMBOLIC_DUPLICATES_LAUNCH
}
