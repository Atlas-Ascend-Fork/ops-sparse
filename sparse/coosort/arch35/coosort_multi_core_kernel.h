/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef COOSORT_MULTI_CORE_KERNEL_H_
#define COOSORT_MULTI_CORE_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "coosort_tiling_data.h"

using namespace AscendC;

/// AscendC/MTE 数据块大小，单位为字节。A32B() 会向上取整元素数量，
/// 保证对应的数据字节数是 32 字节数据块的整数倍。
constexpr uint32_t kBlkSz = 32;

/// GM workspace 中每条记录包含的 int32 字段数：[row, col, 原始位置 P]。
constexpr uint32_t kTrpl = 3;

/// GM 中一条 COO 三元组占 12 字节，每 8 条记录正好是 96 字节，满足
/// 32 字节 MTE 地址对齐要求。
constexpr uint32_t kMergeAlignRecords = 8;

/// co-rank 给出的 A/B 起点不一定按 8 条记录对齐。搬运时 A、B 各自最多
/// 向前附带 7 条前缀，B 在 UB 中的起点向上对齐时还可能产生 7 条空隙，
/// 因此最多额外占 21 条；向上按 8 条记录对齐后统一预留 24 条。
constexpr uint32_t kMergeInputReserveRecords = 3 * kMergeAlignRecords;

/// co-rank 边界比较需要分别缓存左右输入的一个对齐窗口。
constexpr uint32_t kMergePartitionWindowCount = 2;

/// DataCopyExtParams 单个 blockLen 可表达的最大字节数。
constexpr uint32_t kMteMaxBlockBytes = 65535;

/// 升序有符号 int32 Sort 的对齐填充值。使用 INT32_MAX 可以让仅用于对齐的
/// 补齐元素停留在有效 key 之后，避免混入有效排序结果。
constexpr int32_t kPadS = 0x7FFFFFFF;

namespace mc {
template <typename T>
__aicore__ inline uint32_t A32B(uint32_t c)
{
    constexpr uint32_t e = kBlkSz / sizeof(T);
    return (c + e - 1U) & ~(e - 1U);
}
__aicore__ inline void PadT(LocalTensor<int32_t> &t, uint32_t n, uint32_t a, int32_t v)
{
    for (uint32_t i = n; i < a; i++)
        t.SetValue(i, v);
    auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
}
__aicore__ inline bool LEQ(int32_t rA, int32_t cA, int32_t rB, int32_t cB, bool bR)
{
    int32_t pA = bR ? rA : cA, pB = bR ? rB : cB;
    if (pA < pB) {
        return true;
    }
    if (pA > pB) {
        return false;
    }
    int32_t sA = bR ? cA : rA, sB = bR ? cB : rB;
    return sA <= sB;
}
}  // namespace mc

struct CoosortMultiCoreSort {
    __aicore__ inline CoosortMultiCoreSort()
    {}
    __aicore__ inline void Init(
        GM_ADDR r, GM_ADDR c, GM_ADDR pO, GM_ADDR ws, const CoosortTilingData &td, TPipe *pp);
    __aicore__ inline void Process();
    __aicore__ inline void P1Sort(uint32_t tn, int64_t off);
    __aicore__ inline void P1In(uint32_t tn, int64_t off);
    __aicore__ inline void P1Srt(LocalTensor<int32_t> rL, LocalTensor<int32_t> cL, uint32_t n, bool bR,
        LocalTensor<int32_t> oR, LocalTensor<int32_t> oC, LocalTensor<uint32_t> &pL);
    __aicore__ inline void P1Out(uint32_t tn, int64_t off, LocalTensor<int32_t> tr);
    __aicore__ inline void InitMergeBuffers();
    __aicore__ inline void P2Rnd();
    __aicore__ inline uint64_t P2Partition(
        uint64_t aBase, uint64_t aLen, uint64_t bBase, uint64_t bLen, uint64_t diagonal);
    __aicore__ inline void P2MergeSegment(
        uint64_t pairBase, uint64_t aLen, uint64_t bLen, uint64_t diagonalBegin, uint64_t diagonalEnd);
    __aicore__ inline bool P2LessEqual(uint64_t lhs, uint64_t rhs);
    __aicore__ inline void P3Out();

    GlobalTensor<int32_t> gR_, gC_, gP_, gW_[2], gWI_, gWO_;
    TQue<TPosition::VECIN, 1> iR_, iC_;
    TBuf<TPosition::VECCALC> sR_, sC_, sP_;
    TBuf<TPosition::VECCALC> k1_, i1_, g_, b_;
    TBuf<TPosition::VECCALC> sT_;
    TQue<TPosition::VECOUT, 1> o1_;
    TQue<TPosition::VECIN, 1> mI_;
    TQue<TPosition::VECOUT, 1> mO_;
    TBuf<TPosition::VECCALC> pB_;

    TPipe *pp_;
    uint32_t id_ = 0, td_ = 0, rc_ = 0, fc_ = 0, nz_ = 0, aI_ = 0;
    bool bR_ = true;
    uint32_t lN_{0};
    uint64_t cE_{0};
    int32_t oM_{0};
    uint32_t wF_ = 0;
};

__aicore__ inline void CoosortMultiCoreSort::Init(
    GM_ADDR r, GM_ADDR c, GM_ADDR pO, GM_ADDR ws, const CoosortTilingData &td, TPipe *pp)
{
    id_ = GetBlockIdx();
    pp_ = pp;
    nz_ = td.nnz;
    bR_ = td.sortByRow != 0U;
    td_ = td.tileSize;
    rc_ = td.runCount;
    fc_ = td.coreNum;
    oM_ = td.mergeSize;
    gR_.SetGlobalBuffer((__gm__ int32_t *)r);
    gC_.SetGlobalBuffer((__gm__ int32_t *)c);
    gP_.SetGlobalBuffer((__gm__ int32_t *)pO);
    uint64_t h = static_cast<uint64_t>(nz_) * kTrpl;
    uint64_t alignedH = (h + 7U) & ~static_cast<uint64_t>(7U);
    gW_[0].SetGlobalBuffer((__gm__ int32_t *)ws, h);
    gW_[1].SetGlobalBuffer((__gm__ int32_t *)ws + alignedH, h);
    aI_ = mc::A32B<int32_t>(td_);
    pp_->InitBuffer(iR_, 1, aI_ * sizeof(int32_t));
    pp_->InitBuffer(iC_, 1, aI_ * sizeof(int32_t));
    pp_->InitBuffer(sR_, aI_ * sizeof(int32_t));
    pp_->InitBuffer(sC_, aI_ * sizeof(int32_t));
    pp_->InitBuffer(sP_, aI_ * sizeof(uint32_t));
    pp_->InitBuffer(k1_, aI_ * sizeof(int32_t));
    pp_->InitBuffer(i1_, aI_ * sizeof(uint32_t));
    pp_->InitBuffer(g_, aI_ * sizeof(int32_t));
    pp_->InitBuffer(b_, aI_ * sizeof(uint32_t));
    pp_->InitBuffer(sT_, td.sortTmpSize);
    pp_->InitBuffer(o1_, 1, aI_ * kTrpl * sizeof(int32_t));
}

__aicore__ inline void CoosortMultiCoreSort::Process()
{
    if (nz_ == 0) {
        return;
    }
    for (uint32_t runId = id_; runId < rc_; runId += fc_) {
        int64_t o = static_cast<int64_t>(td_) * runId;
        uint32_t remain = nz_ - static_cast<uint32_t>(o);
        uint32_t tn = remain < td_ ? remain : td_;
        P1Sort(tn, o);
    }
    auto phaseOneEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(phaseOneEv);
    WaitFlag<HardEvent::MTE3_S>(phaseOneEv);
    SyncAll();
    pp_->Reset();
    InitMergeBuffers();
    lN_ = rc_;
    cE_ = td_;
    while (lN_ > 1) {
        gWI_ = gW_[wF_];
        gWO_ = gW_[1 - wF_];
        P2Rnd();
        lN_ = (lN_ + 1U) / 2U;
        cE_ *= 2U;
        wF_ = (wF_ + 1U) % 2U;
        pp_->Reset();
        InitMergeBuffers();
    }
    gWI_ = gW_[wF_];
    P3Out();
}

__aicore__ inline void CoosortMultiCoreSort::InitMergeBuffers()
{
    uint32_t mergeTileRecords = static_cast<uint32_t>(oM_);
    uint32_t mergeInputRecords = mergeTileRecords + kMergeInputReserveRecords;
    pp_->InitBuffer(mI_, 1, mergeInputRecords * kTrpl * sizeof(int32_t));
    pp_->InitBuffer(mO_, 1, mergeTileRecords * kTrpl * sizeof(int32_t));
    pp_->InitBuffer(o1_, 1, aI_ * kTrpl * sizeof(int32_t));
    pp_->InitBuffer(pB_, kMergePartitionWindowCount * kMergeAlignRecords * kTrpl * sizeof(int32_t));
}

__aicore__ inline void CoosortMultiCoreSort::P1Sort(uint32_t tn, int64_t off)
{
    P1In(tn, off);
    LocalTensor<int32_t> rL = iR_.DeQue<int32_t>(), cL = iC_.DeQue<int32_t>();
    LocalTensor<int32_t> sR = sR_.Get<int32_t>(), sC = sC_.Get<int32_t>();
    LocalTensor<uint32_t> sP = sP_.Get<uint32_t>(), pL;
    P1Srt(rL, cL, tn, bR_, sR, sC, pL);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> pI = sP.ReinterpretCast<int32_t>();
    for (uint32_t i = 0; i < tn; i++)
        pI.SetValue(i, (int32_t)off + (int32_t)pL.GetValue(i));
    auto ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(ev);
    WaitFlag<HardEvent::S_V>(ev);
    uint32_t al = mc::A32B<int32_t>(tn);
    LocalTensor<int32_t> tr = o1_.AllocTensor<int32_t>();
    // workspace 使用交错三元组布局，使每个有序 run 都是连续记录流：
    // [row, col, 原始下标 P]。
    for (uint32_t i = 0; i < tn; i++) {
        tr.SetValue(i * kTrpl, sR.GetValue(i));
        tr.SetValue(i * kTrpl + 1, sC.GetValue(i));
        tr.SetValue(i * kTrpl + 2, pI.GetValue(i));
    }
    auto packEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(packEv);
    WaitFlag<HardEvent::S_MTE3>(packEv);
    P1Out(tn, off, tr);
    iR_.FreeTensor(rL);
    iC_.FreeTensor(cL);
}

__aicore__ inline void CoosortMultiCoreSort::P1In(uint32_t tn, int64_t off)
{
    DataCopyExtParams cp{1, static_cast<uint32_t>(tn * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
    LocalTensor<int32_t> r = iR_.AllocTensor<int32_t>();
    DataCopyPad(r, gR_[off], cp, pp);
    iR_.EnQue(r);
    LocalTensor<int32_t> c = iC_.AllocTensor<int32_t>();
    DataCopyPad(c, gC_[off], cp, pp);
    iC_.EnQue(c);
}

__aicore__ inline void CoosortMultiCoreSort::P1Srt(LocalTensor<int32_t> rL, LocalTensor<int32_t> cL, uint32_t n,
    bool bR, LocalTensor<int32_t> oR, LocalTensor<int32_t> oC, LocalTensor<uint32_t> &pL)
{
    using namespace mc;
    LocalTensor<int32_t> pT = bR ? rL : cL, sT = bR ? cL : rL;
    LocalTensor<int32_t> dk1 = k1_.Get<int32_t>();
    LocalTensor<uint32_t> di1 = i1_.Get<uint32_t>();
    LocalTensor<uint32_t> ib = b_.Get<uint32_t>();
    LocalTensor<int32_t> pg = g_.Get<int32_t>();
    LocalTensor<uint8_t> st = sT_.Get<uint8_t>();
    uint32_t al = A32B<int32_t>(n);
    PadT(sT, n, al, kPadS);
    static constexpr SortConfig sc{SortType::RADIX_SORT, false};
    Sort<int32_t, false, sc>(dk1, di1, sT, st, n);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> di1s = sP_.Get<uint32_t>();
    DataCopy(di1s, di1, A32B<uint32_t>(n));
    PipeBarrier<PIPE_V>();
    ShiftLeft<uint32_t>(ib, di1s, 2U, (int32_t)n);
    PipeBarrier<PIPE_V>();
    Gather<int32_t>(pg, pT, ib, 0U, n);
    PadT(pg, n, al, kPadS);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> dk2 = k1_.Get<int32_t>();
    LocalTensor<uint32_t> di2 = i1_.Get<uint32_t>();
    Sort<int32_t, false, sc>(dk2, di2, pg, st, n);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> i2b = b_.Get<uint32_t>();
    ShiftLeft<uint32_t>(i2b, di2, 2U, (int32_t)n);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> pU = i1_.Get<uint32_t>();
    Gather<uint32_t>(pU, di1s, i2b, 0U, n);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> pb = b_.Get<uint32_t>();
    ShiftLeft<uint32_t>(pb, pU, 2U, (int32_t)n);
    PipeBarrier<PIPE_V>();
    Gather<int32_t>(oR, rL, pb, 0U, n);
    Gather<int32_t>(oC, cL, pb, 0U, n);
    pL = pU;
}

__aicore__ inline void CoosortMultiCoreSort::P1Out(uint32_t tn, int64_t off, LocalTensor<int32_t> tr)
{
    // P1Out 接管从 o1_ 分配的 Tensor，并在完成 EnQue/DeQue 流程后释放。
    o1_.EnQue(tr);
    LocalTensor<int32_t> o = o1_.DeQue<int32_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(tn * kTrpl * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(gW_[0][off * kTrpl], o, cp);
    o1_.FreeTensor(o);
}

__aicore__ inline void CoosortMultiCoreSort::P2Rnd()
{
    // 两路 merge-path 将本轮完整输出空间切成相互独立的 UB 安全片段，
    // 由各核跨步领取；即使只剩一对 run，也可由多个核并行处理，而不是
    // 固定交给 0 号核串行归并。实际参与核数取决于本轮片段数量。
    // co-rank 边界可能不对齐，因此输入搬运会向前附带少量前缀；一条 COO
    // 三元组为 12 字节，每 8 条共 96 字节，可保持 GM/UB 的 32 字节地址对齐。
    uint64_t tile = static_cast<uint32_t>(oM_);
    uint64_t mteSafeTile = (kMteMaxBlockBytes / (kTrpl * sizeof(int32_t)) - (kMergeAlignRecords - 1U));
    mteSafeTile = (mteSafeTile / kMergeAlignRecords) * kMergeAlignRecords;
    if (tile > mteSafeTile) {
        tile = mteSafeTile;
    }
    tile = (tile / kMergeAlignRecords) * kMergeAlignRecords;
    uint64_t stride = tile * fc_;
    for (uint64_t tileBegin = tile * id_; tileBegin < nz_; tileBegin += stride) {
        uint64_t tileEnd = tileBegin + tile;
        if (tileEnd > nz_) {
            tileEnd = nz_;
        }
        uint64_t pos = tileBegin;
        uint64_t pairSpan = cE_ * 2U;
        while (pos < tileEnd) {
            uint64_t pairBase = (pos / pairSpan) * pairSpan;
            uint64_t aLen = (pairBase + cE_ <= nz_) ? cE_ : static_cast<uint64_t>(nz_) - pairBase;
            uint64_t bBase = pairBase + cE_;
            uint64_t bLen = (bBase < nz_) ? (((bBase + cE_) <= nz_) ? cE_ : static_cast<uint64_t>(nz_) - bBase) : 0U;
            uint64_t pairEnd = pairBase + aLen + bLen;
            uint64_t segmentEnd = (tileEnd < pairEnd) ? tileEnd : pairEnd;
            P2MergeSegment(pairBase, aLen, bLen, pos - pairBase, segmentEnd - pairBase);
            pos = segmentEnd;
        }
    }
    auto mte3Mte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2Ev);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2Ev);
    auto mte3ScalarEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(mte3ScalarEv);
    WaitFlag<HardEvent::MTE3_S>(mte3ScalarEv);
    SyncAll();
}

__aicore__ inline bool CoosortMultiCoreSort::P2LessEqual(uint64_t lhs, uint64_t rhs)
{
    constexpr uint32_t kSlotInts = kMergeAlignRecords * kTrpl;
    uint32_t lhsPrefix = static_cast<uint32_t>(lhs & (kMergeAlignRecords - 1U));
    uint32_t rhsPrefix = static_cast<uint32_t>(rhs & (kMergeAlignRecords - 1U));
    uint64_t lhsBase = lhs - lhsPrefix, rhsBase = rhs - rhsPrefix;
    LocalTensor<int32_t> local = pB_.Get<int32_t>();
    auto scalarMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(scalarMte2Ev);
    WaitFlag<HardEvent::S_MTE2>(scalarMte2Ev);
    DataCopyExtParams lhsCp{1, static_cast<uint32_t>((lhsPrefix + 1U) * kTrpl * sizeof(int32_t)), 0, 0, 0};
    DataCopyExtParams rhsCp{1, static_cast<uint32_t>((rhsPrefix + 1U) * kTrpl * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(local, gWI_[lhsBase * kTrpl], lhsCp, pad);
    DataCopyPad(local[kSlotInts], gWI_[rhsBase * kTrpl], rhsCp, pad);
    auto mte2ScalarEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ScalarEv);
    WaitFlag<HardEvent::MTE2_S>(mte2ScalarEv);
    uint32_t lo = lhsPrefix * kTrpl, ro = kSlotInts + rhsPrefix * kTrpl;
    return mc::LEQ(local.GetValue(lo), local.GetValue(lo + 1U), local.GetValue(ro), local.GetValue(ro + 1U), bR_);
}

__aicore__ inline uint64_t CoosortMultiCoreSort::P2Partition(
    uint64_t aBase, uint64_t aLen, uint64_t bBase, uint64_t bLen, uint64_t diagonal)
{
    uint64_t low = (diagonal > bLen) ? diagonal - bLen : 0U;
    uint64_t high = (diagonal < aLen) ? diagonal : aLen;
    while (low <= high) {
        uint64_t aTake = low + (high - low) / 2U;
        uint64_t bTake = diagonal - aTake;
        if (aTake > 0U && bTake < bLen && !P2LessEqual(aBase + aTake - 1U, bBase + bTake)) {
            high = aTake - 1U;
        } else if (bTake > 0U && aTake < aLen && P2LessEqual(aBase + aTake, bBase + bTake - 1U)) {
            // B[j-1] >= A[i]：增大左分区中的 A 数量；相等时 A 先于 B，
            // 从而保持相邻源 run 之间的稳定性。
            low = aTake + 1U;
        } else {
            return aTake;
        }
    }
    return low;
}

__aicore__ inline void CoosortMultiCoreSort::P2MergeSegment(
    uint64_t pairBase, uint64_t aLen, uint64_t bLen, uint64_t diagonalBegin, uint64_t diagonalEnd)
{
    uint64_t bBase = pairBase + aLen;
    uint64_t aBegin = P2Partition(pairBase, aLen, bBase, bLen, diagonalBegin);
    uint64_t aEnd = P2Partition(pairBase, aLen, bBase, bLen, diagonalEnd);
    uint64_t bBegin = diagonalBegin - aBegin, bEnd = diagonalEnd - aEnd;
    uint32_t aCount = static_cast<uint32_t>(aEnd - aBegin);
    uint32_t bCount = static_cast<uint32_t>(bEnd - bBegin);
    uint32_t outCount = aCount + bCount;
    if (outCount == 0U) {
        return;
    }

    uint32_t aPrefix = static_cast<uint32_t>(aBegin & (kMergeAlignRecords - 1U));
    uint32_t bPrefix = static_cast<uint32_t>(bBegin & (kMergeAlignRecords - 1U));
    uint32_t aCopyCount = aCount + aPrefix;
    uint32_t bCopyCount = bCount + bPrefix;
    uint32_t bLocalBase = (aCopyCount + kMergeAlignRecords - 1U) & ~(kMergeAlignRecords - 1U);
    LocalTensor<int32_t> ub = mI_.AllocTensor<int32_t>();
    if (aCount > 0U) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(aCopyCount * kTrpl * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
        DataCopyPad(ub, gWI_[(pairBase + aBegin - aPrefix) * kTrpl], cp, pp);
    }
    if (bCount > 0U) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(bCopyCount * kTrpl * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
        DataCopyPad(ub[bLocalBase * kTrpl], gWI_[(bBase + bBegin - bPrefix) * kTrpl], cp, pp);
    }
    mI_.EnQue(ub);
    LocalTensor<int32_t> uo = mO_.AllocTensor<int32_t>();
    ub = mI_.DeQue<int32_t>();
    auto mte2ScalarEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ScalarEv);
    WaitFlag<HardEvent::MTE2_S>(mte2ScalarEv);
    uint32_t ai = 0U, bi = 0U;
    for (uint32_t out = 0U; out < outCount; out++) {
        bool takeA = false;
        if (ai < aCount) {
            if (bi >= bCount) {
                takeA = true;
            } else {
                takeA = mc::LEQ(ub.GetValue((aPrefix + ai) * kTrpl),
                    ub.GetValue((aPrefix + ai) * kTrpl + 1U),
                    ub.GetValue((bLocalBase + bPrefix + bi) * kTrpl),
                    ub.GetValue((bLocalBase + bPrefix + bi) * kTrpl + 1U),
                    bR_);
            }
        }
        uint32_t src = takeA ? (aPrefix + ai++) : (bLocalBase + bPrefix + bi++);
        uo.SetValue(out * kTrpl, ub.GetValue(src * kTrpl));
        uo.SetValue(out * kTrpl + 1U, ub.GetValue(src * kTrpl + 1U));
        uo.SetValue(out * kTrpl + 2U, ub.GetValue(src * kTrpl + 2U));
    }
    auto scalarMte3Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(scalarMte3Ev);
    WaitFlag<HardEvent::S_MTE3>(scalarMte3Ev);
    mO_.EnQue(uo);
    mI_.FreeTensor(ub);
    LocalTensor<int32_t> um = mO_.DeQue<int32_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(outCount * kTrpl * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(gWO_[(pairBase + diagonalBegin) * kTrpl], um, cp);
    mO_.FreeTensor(um);
}

__aicore__ inline void CoosortMultiCoreSort::P3Out()
{
    uint32_t chunk = aI_;
    uint32_t outCapacity = static_cast<uint32_t>(oM_);
    if (chunk > outCapacity) {
        chunk = outCapacity;
    }
    if (chunk == 0U) {
        SyncAll();
        return;
    }
    uint64_t stride = static_cast<uint64_t>(chunk) * fc_;
    for (uint64_t elemPos = static_cast<uint64_t>(id_) * chunk; elemPos < nz_; elemPos += stride) {
        uint32_t ce = ((elemPos + chunk) <= nz_) ? chunk : static_cast<uint32_t>(nz_ - elemPos);
        uint32_t c = ce * kTrpl;
        LocalTensor<int32_t> ub = o1_.AllocTensor<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pp{false, 0, 0, 0};
        DataCopyPad(ub, gWI_[elemPos * kTrpl], cp, pp);
        auto mte2ScalarEv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ScalarEv);
        WaitFlag<HardEvent::MTE2_S>(mte2ScalarEv);
        DataCopyExtParams op{1, static_cast<uint32_t>(ce * sizeof(int32_t)), 0, 0, 0};
        uint32_t ace = mc::A32B<int32_t>(ce);
        LocalTensor<int32_t> out = mO_.AllocTensor<int32_t>();
        LocalTensor<int32_t> outR = out;
        LocalTensor<int32_t> outC = out[ace];
        LocalTensor<int32_t> outP = out[ace * 2];
        for (uint32_t i = 0; i < ce; i++) {
            outR.SetValue(i, ub.GetValue(i * kTrpl));
            outC.SetValue(i, ub.GetValue(i * kTrpl + 1));
            outP.SetValue(i, ub.GetValue(i * kTrpl + 2));
        }
        auto scalarMte3Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarMte3Ev);
        WaitFlag<HardEvent::S_MTE3>(scalarMte3Ev);
        mO_.EnQue(out);
        out = mO_.DeQue<int32_t>();
        outR = out;
        outC = out[ace];
        outP = out[ace * 2];
        DataCopyPad(gR_[elemPos], outR, op);
        DataCopyPad(gC_[elemPos], outC, op);
        DataCopyPad(gP_[elemPos], outP, op);
        mO_.FreeTensor(out);
        o1_.FreeTensor(ub);
    }
    SyncAll();
}

extern "C" __global__ __aicore__ void coosort_multi_core_kernel(
    GM_ADDR row, GM_ADDR col, GM_ADDR pOut, GM_ADDR workspace, const CoosortTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CoosortMultiCoreSort op;
    TPipe pp;
    op.Init(row, col, pOut, workspace, tiling, &pp);
    op.Process();
}

extern "C" void coosort_multi_core_kernel_do(GM_ADDR row, GM_ADDR col, GM_ADDR pOut, GM_ADDR workspace,
    const CoosortTilingData &tiling, uint32_t numBlocks, void *stream)
{
    coosort_multi_core_kernel<<<numBlocks, nullptr, stream>>>(row, col, pOut, workspace, tiling);
}

#endif
