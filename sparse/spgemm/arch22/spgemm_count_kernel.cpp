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
 * \file spgemm_count_kernel.cpp
 * \brief SpGEMM arch22 中间乘积计数 Kernel（WorkEstimation 阶段）。
 *
 * 计算 rowProducts[r] = Σ_{k ∈ A.cols(r)} nnz(B.row(k))，即行 r 的中间乘积数 P_i。
 * 该值是分核负载均衡与 nnz(C) 上界估算的依据。
 *
 * 只读 A.rowOffsets / A.colIndices / B.rowOffsets，**不读任何 values**，
 * 故与 dtype 完全无关，四种数据类型共用本 Kernel。
 * 使用 int64 累加，避免高膨胀场景（P_i 可达 nnz(B)）下的 int32 溢出。
 */

#include "kernel_operator.h"
#include "spgemm.h"

using namespace AscendC;

namespace {

// A 行列索引的 UB 分块长度：按块搬入避免 UB 溢出。
constexpr uint32_t kColTile = 2048;
// 小行直读阈值：行 nnz ≤ 该值时直接从 GM 标量读列索引，避免 DataCopyPad 的固定开销。
constexpr uint32_t kSmallRowDirect = 64;
// rowProducts 的回写分块（int64 元素数）。
constexpr uint32_t kOutTile = SPGEMM_ARCH22_ROW_TILE;
// B 行长统计交换区的每核跨距（int32 个数）。64B = 一整条 cacheline，
// 保证按 SINGLE_CACHE_LINE 回刷时各核互不干扰。
constexpr uint32_t kStatStride = 16;

// bApStride 的向量瓦片：一次判定 rows 行，rows·L <= kVecTile 个列索引。
constexpr uint32_t kVecTile = 2048;
// 瓦片行数上限。
constexpr uint32_t kVecRowsMax = kOutTile;
// vecBuf_ 视图偏移（int32 元素），全部 8 元素（32B）对齐。
constexpr uint32_t kVOffCol = 0u * kVecTile;   // 列索引原始瓦片
constexpr uint32_t kVOffHead = 1u * kVecTile;  // col[r·L]（行内广播）
constexpr uint32_t kVOffNext = 2u * kVecTile;  // col[r·L+1] → 公差 → 判定值
constexpr uint32_t kVOffChk = 3u * kVecTile;   // 差值（int32）
constexpr uint32_t kVOffChkF = 4u * kVecTile;  // 差值（float 视图，供 Abs/Reduce）
constexpr uint32_t kVOffWork = 5u * kVecTile;  // ReduceMax 的临时缓冲
constexpr uint32_t kVOffIdxH = 6u * kVecTile;  // Gather 偏移表：r·L
constexpr uint32_t kVOffIdxN = 7u * kVecTile;  // Gather 偏移表：r·L+1
constexpr uint32_t kVOffMIdx = 8u * kVecTile;   // m = j%L
constexpr uint32_t kVOffLayer = 9u * kVecTile;  // layer = j/L
constexpr uint32_t kVOffJBase = 10u * kVecTile; // jBase = j（与 L 无关）
constexpr uint32_t kVSmall = 11u * kVecTile;
// 小表区：每块 384 个元素（1536B，32B 对齐），足以容纳 kVecRowsMax+1 个元素并留出对齐余量。
constexpr uint32_t kVOffPtr = kVSmall + 0u;      // bRowPtr 瓦片（rows+1）
constexpr uint32_t kVOffRamp = kVSmall + 384u;   // r·L（rows+1）
constexpr uint32_t kVOffRowH = kVSmall + 768u;   // 行级 Gather 偏移：4·r·L
constexpr uint32_t kVOffRowN = kVSmall + 1152u;  // 行级 Gather 偏移：4·r·L+4
constexpr uint32_t kVOffRed = kVSmall + 1536u;   // ReduceMax 结果
constexpr uint32_t kVecBufElems = kVSmall + 1920u;

// gcd(L, 8)：只有 1/2/4/8 四种取值，避免在 Kernel 里跑辗转相除。
__aicore__ inline uint32_t Gcd8(uint32_t l)
{
    if ((l & 7u) == 0u) {
        return 8u;
    }
    if ((l & 3u) == 0u) {
        return 4u;
    }
    return ((l & 1u) == 0u) ? 2u : 1u;
}

class SpgemmArch22CountKernel {
public:
    __aicore__ inline void Init(GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR bRowPtr,
                                GM_ADDR bColIdx, GM_ADDR buffer1, GM_ADDR tilingGm)
    {
        auto *td = reinterpret_cast<__gm__ SpgemmArch22TilingData *>(tilingGm);
        M_ = td->M;
        nnzA_ = td->nnzA;
        nnzB_ = td->nnzB;
        K_ = td->K;

        const uint32_t blockDim = GetBlockNum();
        const uint32_t blockId = GetBlockIdx();
        if (blockDim == 0) return;
        // WorkEstimation 阶段 binEdge 尚未生成，按行数均分即可（此阶段每行成本
        // 正比于 A 行 nnz，均分已足够；真正的负载均衡在 binEdge 生成后生效）。
        const uint32_t rowsPerCore = (M_ + blockDim - 1) / blockDim;
        rowStart_ = blockId * rowsPerCore;
        rowEnd_ = rowStart_ + rowsPerCore;
        if (rowEnd_ > M_) { rowEnd_ = M_; }
        if (rowStart_ > M_) { rowStart_ = M_; }
        // bApStride 的分核：按 B 行数均分。每核成本正比于其区间内的 nnz(B)，
        // B 行长在实际负载里高度均匀（且本趟总成本只有 nnz(B) 次读），均分即可。
        const uint32_t kPerCore = (K_ + blockDim - 1) / blockDim;
        kStart_ = blockId * kPerCore;
        kEnd_ = kStart_ + kPerCore;
        if (kEnd_ > K_) { kEnd_ = K_; }
        if (kStart_ > K_) { kStart_ = K_; }

        aRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aRowPtr), M_ + 1);
        aColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aColIdx), nnzA_);
        bRowPtrGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bRowPtr), K_ + 1);
        bColIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bColIdx), nnzB_);
        rowProductsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(buffer1 + td->rowProductsOffset), M_);
        bApStrideGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bApStrideOffset),
            (K_ > 0) ? K_ : 1);
        bHeadGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bHeadOffset),
            (K_ > 0) ? K_ : 1);
        bStatGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->bStatOffset),
            blockDim * kStatStride);
        syncGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(buffer1 + td->syncOffset), blockDim * 8);
        blockDim_ = blockDim;

        pipe_.InitBuffer(colQue_, 1, kColTile * sizeof(int32_t));
        pipe_.InitBuffer(outQue_, 1, kOutTile * sizeof(int64_t));
        pipe_.InitBuffer(apQue_, 1, kOutTile * sizeof(int32_t));
        pipe_.InitBuffer(headQue_, 1, kOutTile * sizeof(int32_t));
        // Software SyncAll stages a 32-byte flag slot for each participating core in local UB.
        pipe_.InitBuffer(syncQue_, 1, blockDim * 8 * sizeof(int32_t));
        // 向量瓦片工作区。
        pipe_.InitBuffer(vecBuf_, kVecBufElems * sizeof(int32_t));
    }

    /**
     * 执行顺序：先建 B 的行描述符并统计全局行长 min/max，
     * 跨核归约后再算 rowProducts。
     */
    __aicore__ inline void Process()
    {
        BuildBApStride();
        PublishBStat();
        LocalTensor<int32_t> syncLocal = syncQue_.AllocTensor<int32_t>();
        SyncAll(syncGm_, syncLocal, static_cast<int32_t>(blockDim_));
        syncQue_.FreeTensor(syncLocal);
        LoadBStat();
        CountRowProducts();
    }

private:
    /**
     * 把本核区间内的 B 行长 min/max 写到交换区。空区间写中性初值。
     *
     * 每核独占 kStatStride 个 int32（= 64B，一整条 cacheline），
     * 使用 SINGLE_CACHE_LINE 回刷避免伪共享。
     */
    __aicore__ inline void PublishBStat()
    {
        const uint32_t slot = GetBlockIdx() * kStatStride;
        bStatGm_.SetValue(slot, minLen_);
        bStatGm_.SetValue(slot + 1, maxLen_);
        // 槽位 2 = 本核最小的「带外」行号（全带内则 INT32_MAX）。
        bStatGm_.SetValue(slot + 2, static_cast<int32_t>(bandBad_));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                DcciDst::CACHELINE_OUT>(bStatGm_[slot]);
    }

    /**
     * 跨核归约：全局 min == max 且 > 0 ⇒ B 每行等长，记 L = 该值。
     * 读前对每条槽位做一次回刷失效，强制从 GM 重取。
     */
    __aicore__ inline void LoadBStat()
    {
        int32_t gMin = 0x7FFFFFFF;
        int32_t gMax = -0x7FFFFFFF;
        for (uint32_t c = 0; c < blockDim_; c++) {
            const uint32_t slot = c * kStatStride;
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
        }
        bUniformLen_ = ((K_ > 0) && (gMin == gMax) && (gMin > 0)) ? gMin : 0;
    }

    /**
     * B 每行等差公差描述符。
     *
     * bApStride[k] = d 当且仅当 L = nnz(B.row k) >= 2 且该行列索引恰为公差 d 的
     * 等差数列；否则 0。d <= 0 时如实写入，由下游 `stride > 0` 过滤。
     */
    /**
     * 单行的等差校验。c0/c1 由调用方预先读好传入，本函数
     * 只做尾部 L-2 个元素的校验。
     */
    __aicore__ inline int32_t ApTail(int32_t bs, int32_t be, int32_t c0, int32_t c1)
    {
        const int32_t d = c1 - c0;
        // 无分支累加：把「读→比较→分支」换成「读→XOR→或」，减少循环携带依赖。
        uint32_t diff = 0;
        uint32_t expect = static_cast<uint32_t>(c1);
        const uint32_t st = static_cast<uint32_t>(d);
        for (int32_t p = bs + 2; p < be; p++) {
            expect += st;
            diff |= static_cast<uint32_t>(
                        bColIdxGm_.GetValue(static_cast<uint64_t>(p))) ^ expect;
        }
        return (diff == 0) ? d : 0;
    }

    /**
     * 从 B 的第 0 行取「带状矩阵」参考三元组 (L, s, h0)。
     *
     * 带状的含义：bColIdx[bRowPtr[k] + j] == h0 + k + j·s，且每行长度恒为 L。
     * 满足这一形态的行，其描述符可由 k 解析算出，避免二级 GM 读。
     *
     * 参考值取自第 0 行：BuildBApStride 在 PublishBStat/SyncAll 之前执行，
     * 此刻全局 L 还不知道，第 0 行是无需跨核通信即可获取的锚点。
     */
    __aicore__ inline void LoadBandRef()
    {
        bandRefOk_ = false;
        if (K_ == 0) {
            return;
        }
        const int32_t p0 = bRowPtrGm_.GetValue(0);
        const int32_t p1 = bRowPtrGm_.GetValue(1);
        // 带状要求 bRowPtr[0] == 0。L < 2 时无法取 bColIdx[1]，直接放弃。
        if ((p0 != 0) || (p1 < 2)) {
            return;
        }
        bandL_ = p1;
        bandH_ = bColIdxGm_.GetValue(0);
        bandS_ = ApTail(0, bandL_, bandH_, bColIdxGm_.GetValue(1));
        bandRefOk_ = true;
    }

    // 带是**前缀**，故只需记最小的违规行号；后面的行一律视作带外。
    __aicore__ inline void MarkBandBad(uint32_t k)
    {
        if (k < bandBad_) {
            bandBad_ = k;
        }
    }

    /**
     * 标量路径的逐行带校验。rowLen/stride/headCol 来自寄存器，零额外 GM 读。
     */
    __aicore__ inline void CheckBand(uint32_t k, int32_t rowLen, int32_t stride,
                                     int32_t headCol)
    {
        const int64_t want = static_cast<int64_t>(bandH_) + static_cast<int64_t>(k);
        if ((!bandRefOk_) || (rowLen != bandL_) || (stride != bandS_) ||
            (static_cast<int64_t>(headCol) != want)) {
            MarkBandBad(k);
        }
    }

    __aicore__ inline void StatLen(int32_t rowLen)
    {
        if (rowLen < minLen_) {
            minLen_ = rowLen;
        }
        if (rowLen > maxLen_) {
            maxLen_ = rowLen;
        }
    }

    /**
     * 构建 jBase[j] = j 表。与 L 无关，整个 Kernel 只建一次。
     * 用「铺 8 个 + 向量倍增」代替逐元素标量写。
     */
    __aicore__ inline void BuildRampTable()
    {
        const LocalTensor<int32_t> jBase = vecBuf_.Get<int32_t>()[kVOffJBase];
        for (uint32_t j = 0; j < 8u; j++) {
            jBase.SetValue(j, static_cast<int32_t>(j));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
        uint32_t len = 8u;
        while (len < kVecTile) {
            Adds(jBase[len], jBase, static_cast<int32_t>(len), static_cast<int32_t>(len));
            len += len;
        }
        rampOk_ = true;
    }

    /**
     * 按 L 建行不变的索引表（layer/mIdx/idxH/idxN/ramp/rowH/rowN）。
     * L 不变时表可复用，只在 L 变化时重建。
     *
     * @param L  B 行长度
     * @param pat  lcm(L, 8)，用于铺底 + 倍增
     */
    __aicore__ inline void BuildVecTables(int32_t L, uint32_t pat)
    {
        if (!rampOk_) {
            BuildRampTable();
        }
        const LocalTensor<int32_t> pool = vecBuf_.Get<int32_t>();
        const LocalTensor<int32_t> jBase = pool[kVOffJBase];
        const LocalTensor<int32_t> layer = pool[kVOffLayer];
        const LocalTensor<int32_t> mIdx = pool[kVOffMIdx];
        const LocalTensor<int32_t> idxH = pool[kVOffIdxH];
        const LocalTensor<int32_t> idxN = pool[kVOffIdxN];
        const LocalTensor<int32_t> ramp = pool[kVOffRamp];
        const LocalTensor<int32_t> rowH = pool[kVOffRowH];
        const LocalTensor<int32_t> rowN = pool[kVOffRowN];
        const LocalTensor<int32_t> tmp = pool[kVOffChk];
        const uint32_t lu = static_cast<uint32_t>(L);
        for (uint32_t j = 0; j < pat; j++) {
            layer.SetValue(j, static_cast<int32_t>(j / lu));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
        uint32_t len = pat;
        while (len < kVecTile) {
            uint32_t n2 = kVecTile - len;
            if (n2 > len) {
                n2 = len;
            }
            Adds(layer[len], layer, static_cast<int32_t>(len / lu), static_cast<int32_t>(n2));
            len += n2;
        }
        constexpr int32_t kAll = static_cast<int32_t>(kVecTile);
        Muls(tmp, layer, L, kAll);
        Sub(mIdx, jBase, tmp, kAll);                       // m = j − L·(j/L)
        Muls(idxH, layer, 4 * L, kAll);                    // 字节偏移 4·L·(j/L)
        Adds(idxN, idxH, 4, kAll);
        Muls(ramp, jBase, L, static_cast<int32_t>(kVecRowsMax + 1u));
        Muls(rowH, jBase, 4 * L, static_cast<int32_t>(kVecRowsMax));
        Adds(rowN, rowH, 4, static_cast<int32_t>(kVecRowsMax));
        tblL_ = L;
    }

    /**
     * 判定一：瓦片内行长恒为 L（列索引区间连续且长度为 rows·L）。
     */
    __aicore__ inline bool ProbeApVecUniformLen(uint32_t k, int32_t p0, uint32_t rows)
    {
        const LocalTensor<int32_t> pool = vecBuf_.Get<int32_t>();
        const LocalTensor<float> poolF = vecBuf_.Get<float>();
        const LocalTensor<int32_t> chk = pool[kVOffChk];
        const LocalTensor<int32_t> ptrI = pool[kVOffPtr];
        const LocalTensor<int32_t> ramp = pool[kVOffRamp];
        const LocalTensor<float> chkF = poolF[kVOffChkF];
        const LocalTensor<float> work = poolF[kVOffWork];
        const LocalTensor<float> red = poolF[kVOffRed];
        DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        const int32_t cntP = static_cast<int32_t>(rows + 1u);
        DataCopyExtParams cpP{1, static_cast<uint32_t>((rows + 1u) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(ptrI, bRowPtrGm_[k], cpP, padI);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Sub(chk, ptrI, ramp, cntP);
        Adds(chk, chk, -p0, cntP);
        // int32 差值转 float 借用 Abs/ReduceMax 判定是否全零，转换不引入误判。
        Cast(chkF, chk, RoundMode::CAST_NONE, cntP);
        Abs(chkF, chkF, cntP);
        ReduceMax<float>(red, chkF, work, cntP, false);
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        return red.GetValue(0) == 0.0f;
    }

    /**
     * 判定二：每行都是等差数列。col[j] ?= col[r·L] + m·(col[r·L+1] − col[r·L])。
     * 全程 int32，溢出回绕下判定仍精确。顺带把 B 列索引搬进 colI。
     */
    __aicore__ inline bool ProbeApVecArithmetic(int32_t p0, uint32_t n)
    {
        const LocalTensor<int32_t> pool = vecBuf_.Get<int32_t>();
        const LocalTensor<float> poolF = vecBuf_.Get<float>();
        const LocalTensor<int32_t> colI = pool[kVOffCol];
        const LocalTensor<int32_t> headN = pool[kVOffHead];
        const LocalTensor<int32_t> nextN = pool[kVOffNext];
        const LocalTensor<int32_t> chk = pool[kVOffChk];
        const LocalTensor<float> chkF = poolF[kVOffChkF];
        const LocalTensor<float> work = poolF[kVOffWork];
        const LocalTensor<float> red = poolF[kVOffRed];
        // Gather 的偏移张量用 uint32 视图；值 <= 4·kVecTile，int32/uint32 位型一致。
        const LocalTensor<uint32_t> idxH = pool[kVOffIdxH].ReinterpretCast<uint32_t>();
        const LocalTensor<uint32_t> idxN = pool[kVOffIdxN].ReinterpretCast<uint32_t>();
        const LocalTensor<int32_t> mIdx = pool[kVOffMIdx];
        DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        const int32_t cntN = static_cast<int32_t>(n);
        DataCopyExtParams cpC{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(colI, bColIdxGm_[static_cast<uint64_t>(p0)], cpC, padI);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Gather(headN, colI, idxH, 0u, n);
        Gather(nextN, colI, idxN, 0u, n);
        Sub(nextN, nextN, headN, cntN);
        Mul(nextN, nextN, mIdx, cntN);
        Add(nextN, nextN, headN, cntN);
        Sub(chk, nextN, colI, cntN);
        Cast(chkF, chk, RoundMode::CAST_NONE, cntN);
        Abs(chkF, chkF, cntN);
        ReduceMax<float>(red, chkF, work, cntN, false);
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        return red.GetValue(0) == 0.0f;
    }

    /**
     * 判定失败时折半重试，返回可用行数，0 表示回落到标量路径。
     * 折半保持 8 的倍数以满足 Gather 对齐要求。
     */
    __aicore__ inline uint32_t ProbeApVecTile(uint32_t k, int32_t p0, uint32_t lu, uint32_t rows)
    {
        while (rows >= 8u) {
            if (ProbeApVecUniformLen(k, p0, rows) && ProbeApVecArithmetic(p0, rows * lu)) {
                return rows;
            }
            // 折半后保持 8 的倍数以维持 32B 对齐。
            rows = (rows >> 1) & ~7u;
        }
        return 0;
    }

    /**
     * 向量瓦片的带状校验（ap[r] == s 且 head[r] == h0 + k + r）。
     * 溢出护栏：h0 + k + rows 必须在 int32 内，否则判带外。
     */
    __aicore__ inline void CheckApVecBand(uint32_t k, uint32_t rows, int32_t rowLen,
                                          uint32_t pending,
                                          const LocalTensor<int32_t> &apLocal,
                                          const LocalTensor<int32_t> &headLocal)
    {
        if ((bandBad_ > k) && bandRefOk_ && (rowLen == bandL_) &&
            ((static_cast<int64_t>(bandH_) + static_cast<int64_t>(k) +
              static_cast<int64_t>(rows)) <= 0x7FFFFFFF)) {
            const LocalTensor<int32_t> pool = vecBuf_.Get<int32_t>();
            const LocalTensor<float> poolF = vecBuf_.Get<float>();
            const LocalTensor<int32_t> nextN = pool[kVOffNext];
            const LocalTensor<float> chkF = poolF[kVOffChkF];
            const LocalTensor<float> work = poolF[kVOffWork];
            const LocalTensor<float> red = poolF[kVOffRed];
            // 两个判据合成一次归约：|ap − s| 与 |head − h0 − k − r| 的浮点和
            // 为 0 ⟺ 两者同时为 0。
            const int32_t cntR = static_cast<int32_t>(rows);
            const LocalTensor<float> chkF2 = poolF[kVOffHead];  // headN 此刻已死
            const LocalTensor<int32_t> jBase = pool[kVOffJBase];
            Adds(nextN, apLocal[pending], -bandS_, cntR);
            Cast(chkF, nextN, RoundMode::CAST_NONE, cntR);
            Abs(chkF, chkF, cntR);
            // jBase[r] = r（单位斜坡，与 L 无关，不能用 ramp）。
            Adds(nextN, headLocal[pending], -(bandH_ + static_cast<int32_t>(k)), cntR);
            Sub(nextN, nextN, jBase, cntR);
            Cast(chkF2, nextN, RoundMode::CAST_NONE, cntR);
            Abs(chkF2, chkF2, cntR);
            Add(chkF, chkF, chkF2, cntR);
            ReduceMax<float>(red, chkF, work, cntR, false);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            if (red.GetValue(0) != 0.0f) { MarkBandBad(k); }
        } else if (bandBad_ > k) {
            MarkBandBad(k);
        }
    }

    /**
     * 一个向量瓦片。成功返回已写入描述符的行数，0 表示未走本路径（回落到标量）。
     *
     * 仅当瓦片内行长恒为 L >= 2 且每行都是等差数列时才走这里。
     */
    __aicore__ inline uint32_t BuildApVecTile(uint32_t k, uint32_t roomRows, uint32_t pending,
                                             const LocalTensor<int32_t> &apLocal,
                                             const LocalTensor<int32_t> &headLocal)
    {
        // Gather 目的地须 32B 对齐 ⇒ pending 须为 8 的倍数。
        if ((roomRows < 8u) || ((pending & 7u) != 0u)) { return 0; }
        const int32_t p0 = bRowPtrGm_.GetValue(k);
        const int32_t p1 = bRowPtrGm_.GetValue(k + 1);
        const int32_t rowLen = p1 - p0;
        // L < 2：无法取 col[r·L+1]。L 太大：一个瓦片放不下 8 行。
        if ((rowLen < 2) || (static_cast<uint32_t>(rowLen) > kVecTile / 8u)) { return 0; }
        const uint32_t lu = static_cast<uint32_t>(rowLen);
        uint32_t rowsMax = kVecTile / lu;
        if (rowsMax > kVecRowsMax) { rowsMax = kVecRowsMax; }
        rowsMax &= ~7u;
        const uint32_t rowsTry = (roomRows < rowsMax) ? (roomRows & ~7u) : rowsMax;
        if (rowsTry < 8u) { return 0; }
        if (tblL_ != rowLen) {
            // 建表需要 pat = lcm(L, 8) 次标量写，pat 超限则走标量路径。
            const uint32_t g = Gcd8(lu);
            if (g == 0) return 0;
            const uint32_t pat = lu * (8u / g);
            if ((pat > kVecTile) || (pat > (kEnd_ - kStart_))) { return 0; }
            BuildVecTables(rowLen, pat);
        }

        // 两道判定 + 折半重试。
        const uint32_t rows = ProbeApVecTile(k, p0, lu, rowsTry);
        if (rows == 0u) { return 0; }

        // ---- 落描述符：head[r] = col[r·L]，stride[r] = col[r·L+1] − col[r·L]。
        const LocalTensor<int32_t> pool = vecBuf_.Get<int32_t>();
        const LocalTensor<int32_t> colI = pool[kVOffCol];
        const LocalTensor<int32_t> chk = pool[kVOffChk];
        const LocalTensor<uint32_t> rowH = pool[kVOffRowH].ReinterpretCast<uint32_t>();
        const LocalTensor<uint32_t> rowN = pool[kVOffRowN].ReinterpretCast<uint32_t>();
        Gather(headLocal[pending], colI, rowH, 0u, rows);
        Gather(chk, colI, rowN, 0u, rows);
        Sub(apLocal[pending], chk, headLocal[pending], static_cast<int32_t>(rows));
        // 等待向量写完成后才能进行后续标量/MTE3 操作。
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        CheckApVecBand(k, rows, rowLen, pending, apLocal, headLocal);
        StatLen(rowLen);
        return rows;
    }

    __aicore__ inline void BuildBApStride()
    {
        if (kStart_ >= kEnd_) {
            return;
        }
        LoadBandRef();
        LocalTensor<int32_t> apLocal = apQue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> headLocal = headQue_.AllocTensor<int32_t>();
        uint32_t pending = 0;
        uint32_t flushBase = kStart_;

        uint32_t k = kStart_;
        // 连续两组共享一次 bRowPtr 读。走过向量瓦片后 prev 失效。
        int32_t prev = 0;
        bool prevOk = false;
        while (k < kEnd_) {
            uint32_t room = kEnd_ - k;
            if (room > kOutTile - pending) {
                room = kOutTile - pending;
            }
            // ---- 先试向量瓦片。不满足前提时零成本退出。
            const uint32_t got = BuildApVecTile(k, room, pending, apLocal, headLocal);
            if (got != 0u) {
                pending += got;
                k += got;
                prevOk = false;
            } else if (room >= 4u) {
                BuildApGroup4(k, pending, prev, prevOk, apLocal, headLocal);
                prevOk = true;
                pending += 4u;
                k += 4u;
            } else {
                BuildApRow1(k, pending, apLocal, headLocal);
                prevOk = false;
                pending += 1u;
                k += 1u;
            }
            if (pending == kOutTile) {
                FlushAp(apLocal, headLocal, flushBase, pending);
                flushBase = k;
                pending = 0;
            }
        }
        if (pending > 0) {
            FlushAp(apLocal, headLocal, flushBase, pending);
        }
        headQue_.FreeTensor(headLocal);
        apQue_.FreeTensor(apLocal);
    }

    /**
     * 单行的保守路径。
     */
    __aicore__ inline void BuildApRow1(uint32_t k, uint32_t pending,
                                       const LocalTensor<int32_t> &apLocal,
                                       const LocalTensor<int32_t> &headLocal)
    {
        const int32_t bs = bRowPtrGm_.GetValue(k);
        const int32_t be = bRowPtrGm_.GetValue(k + 1);
        StatLen(be - bs);
        int32_t stride = 0;
        int32_t headCol = 0;
        if (be - bs >= 2) {
            const int32_t c0 = bColIdxGm_.GetValue(static_cast<uint64_t>(bs));
            const int32_t c1 = bColIdxGm_.GetValue(static_cast<uint64_t>(bs) + 1);
            headCol = c0;
            stride = ApTail(bs, be, c0, c1);
        } else if (be - bs == 1) {
            headCol = bColIdxGm_.GetValue(static_cast<uint64_t>(bs));
        }
        CheckBand(k, be - bs, stride, headCol);
        apLocal.SetValue(pending, stride);
        headLocal.SetValue(pending, headCol);
    }

    /**
     * 4 行一组，先把 4 行的 c0/c1 全部发射出去，再做校验。
     * 组内 4 行必须全部 rowLen >= 2 才走批量路径，否则逐行走保守路径。
     */
    /**
     * 组内 4 行全部 rowLen >= 2 时的批量描述符构建。
     * 8 笔互不依赖的 bColIdx 读连续发射以提高 MLP。
     */
    __aicore__ inline void BuildApGroup4Uniform(uint32_t k, uint32_t pending,
                                                const int32_t (&bsv)[4], const int32_t (&bev)[4],
                                                const int32_t (&lens)[4],
                                                const LocalTensor<int32_t> &apLocal,
                                                const LocalTensor<int32_t> &headLocal)
    {
        const int32_t a0 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[0]));
        const int32_t b0 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[0]) + 1);
        const int32_t a1 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[1]));
        const int32_t b1 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[1]) + 1);
        const int32_t a2 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[2]));
        const int32_t b2 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[2]) + 1);
        const int32_t a3 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[3]));
        const int32_t b3 = bColIdxGm_.GetValue(static_cast<uint64_t>(bsv[3]) + 1);
        // bHead[k] = 首列索引。
        const int32_t s0 = ApTail(bsv[0], bev[0], a0, b0);
        const int32_t s1 = ApTail(bsv[1], bev[1], a1, b1);
        const int32_t s2 = ApTail(bsv[2], bev[2], a2, b2);
        const int32_t s3 = ApTail(bsv[3], bev[3], a3, b3);
        apLocal.SetValue(pending, s0);
        headLocal.SetValue(pending, a0);
        apLocal.SetValue(pending + 1, s1);
        headLocal.SetValue(pending + 1, a1);
        apLocal.SetValue(pending + 2, s2);
        headLocal.SetValue(pending + 2, a2);
        apLocal.SetValue(pending + 3, s3);
        headLocal.SetValue(pending + 3, a3);
        CheckBand(k, lens[0], s0, a0);
        CheckBand(k + 1, lens[1], s1, a1);
        CheckBand(k + 2, lens[2], s2, a2);
        CheckBand(k + 3, lens[3], s3, a3);
    }

    /**
     * 组内有空行或单元素行时的逐行保守路径。
     */
    __aicore__ inline void BuildApGroup4Scalar(uint32_t k, uint32_t pending,
                                               const int32_t (&bsv)[4], const int32_t (&bev)[4],
                                               const LocalTensor<int32_t> &apLocal,
                                               const LocalTensor<int32_t> &headLocal)
    {
        for (uint32_t j = 0; j < 4; j++) {
            const int32_t bs = bsv[j];
            const int32_t be = bev[j];
            int32_t stride = 0;
            int32_t headCol = 0;
            if (be - bs >= 2) {
                const int32_t c0 = bColIdxGm_.GetValue(static_cast<uint64_t>(bs));
                const int32_t c1 =
                    bColIdxGm_.GetValue(static_cast<uint64_t>(bs) + 1);
                headCol = c0;
                stride = ApTail(bs, be, c0, c1);
            } else if (be - bs == 1) {
                headCol = bColIdxGm_.GetValue(static_cast<uint64_t>(bs));
            }
            CheckBand(k + j, be - bs, stride, headCol);
            apLocal.SetValue(pending + j, stride);
            headLocal.SetValue(pending + j, headCol);
        }
    }

    __aicore__ inline void BuildApGroup4(uint32_t k, uint32_t pending, int32_t &prev, bool prevOk,
                                         const LocalTensor<int32_t> &apLocal,
                                         const LocalTensor<int32_t> &headLocal)
    {
        const int32_t bs0 = prevOk ? prev : bRowPtrGm_.GetValue(k);
        const int32_t bs1 = bRowPtrGm_.GetValue(k + 1);
        const int32_t bs2 = bRowPtrGm_.GetValue(k + 2);
        const int32_t bs3 = bRowPtrGm_.GetValue(k + 3);
        const int32_t be3 = bRowPtrGm_.GetValue(k + 4);
        prev = be3;
        const int32_t l0 = bs1 - bs0;
        const int32_t l1 = bs2 - bs1;
        const int32_t l2 = bs3 - bs2;
        const int32_t l3 = be3 - bs3;
        // 行长统计，零额外 GM 读。
        StatLen(l0);
        StatLen(l1);
        StatLen(l2);
        StatLen(l3);
        int32_t lmin = l0 < l1 ? l0 : l1;
        const int32_t lmin2 = l2 < l3 ? l2 : l3;
        lmin = lmin < lmin2 ? lmin : lmin2;
        const int32_t bsv[4] = {bs0, bs1, bs2, bs3};
        const int32_t bev[4] = {bs1, bs2, bs3, be3};
        if (lmin >= 2) {
            const int32_t lens[4] = {l0, l1, l2, l3};
            BuildApGroup4Uniform(k, pending, bsv, bev, lens, apLocal, headLocal);
        } else {
            // 组内有空行或单元素行：逐行走保守路径。
            BuildApGroup4Scalar(k, pending, bsv, bev, apLocal, headLocal);
        }
    }

    /**
     * 小行快路径的段长累加。
     *
     * 2 路 ILP 展开：相邻段的地址链彼此独立，交错发射可重叠延迟。
     * int64 双 partial 合并，越界列索引跳过。
     */
    __aicore__ inline int64_t CountSmallRowProducts(uint32_t rs, uint32_t re)
    {
        // 小行快路径：直读 GM，省掉搬运启动与流水同步。
        uint32_t i = rs;
        int64_t prod0 = 0;
        int64_t prod1 = 0;
        for (; i + 1 < re; i += 2) {
            const int32_t k0 = aColIdxGm_.GetValue(i);
            const int32_t k1 = aColIdxGm_.GetValue(i + 1);
            const bool ok0 = (k0 >= 0) && (static_cast<uint32_t>(k0) < K_);
            const bool ok1 = (k1 >= 0) && (static_cast<uint32_t>(k1) < K_);
            if (ok0 && ok1) {
                const int32_t bs0 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k0));
                const int32_t bs1 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k1));
                const int32_t be0 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k0) + 1);
                const int32_t be1 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k1) + 1);
                prod0 += (be0 > bs0) ? static_cast<int64_t>(be0 - bs0) : 0;
                prod1 += (be1 > bs1) ? static_cast<int64_t>(be1 - bs1) : 0;
            } else {
                if (ok0) {
                    const int32_t bs0 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k0));
                    const int32_t be0 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k0) + 1);
                    prod0 += (be0 > bs0) ? static_cast<int64_t>(be0 - bs0) : 0;
                }
                if (ok1) {
                    const int32_t bs1 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k1));
                    const int32_t be1 = bRowPtrGm_.GetValue(static_cast<uint32_t>(k1) + 1);
                    prod1 += (be1 > bs1) ? static_cast<int64_t>(be1 - bs1) : 0;
                }
            }
        }
        int64_t products = prod0 + prod1;
        for (; i < re; i++) {
            const int32_t k = aColIdxGm_.GetValue(i);
            if (k < 0 || static_cast<uint32_t>(k) >= K_) {
                continue;
            }
            const int32_t bs = bRowPtrGm_.GetValue(static_cast<uint32_t>(k));
            const int32_t be = bRowPtrGm_.GetValue(static_cast<uint32_t>(k) + 1);
            // 钳到 0：损坏 CSR 下 be < bs 时不计入。
            products += (be > bs) ? static_cast<int64_t>(be - bs) : 0;
        }
        return products;
    }

    /**
     * 长行路径：按 kColTile 分块把 A 行列索引搬进 UB 再累加段长。
     */
    __aicore__ inline int64_t CountRowProductsChunked(uint32_t rs, uint32_t re,
                                                      const LocalTensor<int32_t> &colLocal)
    {
        int64_t products = 0;
        // 按 kColTile 分块搬入 A 行的列索引，逐个查 B 的行长度累加。
        for (uint32_t base = rs; base < re; base += kColTile) {
            uint32_t len = re - base;
            if (len > kColTile) {
                len = kColTile;
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(colLocal, aColIdxGm_[base], cp, pad);
            // 标量读取 UB 前需等待 MTE2 搬运完成。
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            for (uint32_t i = 0; i < len; i++) {
                const int32_t k = colLocal.GetValue(i);
                // 非法列索引（越界）跳过，避免读 B.rowPtr 越界。
                if (k < 0 || static_cast<uint32_t>(k) >= K_) {
                    continue;
                }
                const int32_t bs = bRowPtrGm_.GetValue(static_cast<uint32_t>(k));
                const int32_t be = bRowPtrGm_.GetValue(static_cast<uint32_t>(k) + 1);
                // 钳到 0。
                products += (be > bs) ? static_cast<int64_t>(be - bs) : 0;
            }
        }
        return products;
    }

    __aicore__ inline void CountRowProducts()
    {
        if (rowStart_ >= rowEnd_) {
            return;
        }
        if (bUniformLen_ > 0) {
            CountRowProductsUniform();
            return;
        }
        LocalTensor<int64_t> outLocal = outQue_.AllocTensor<int64_t>();
        LocalTensor<int32_t> colLocal = colQue_.AllocTensor<int32_t>();

        uint32_t pending = 0;         // outLocal 中已填元素数
        uint32_t flushBase = rowStart_;  // outLocal 对应的起始行

        for (uint32_t row = rowStart_; row < rowEnd_; row++) {
            const uint32_t rs = static_cast<uint32_t>(aRowPtrGm_.GetValue(row));
            const uint32_t re = static_cast<uint32_t>(aRowPtrGm_.GetValue(row + 1));
            // 分档与原实现一致：小行直读 GM，长行按 kColTile 分块搬入 UB。
            const int64_t products = (re - rs <= kSmallRowDirect)
                                         ? CountSmallRowProducts(rs, re)
                                         : CountRowProductsChunked(rs, re, colLocal);
            outLocal.SetValue(pending, products);
            pending++;
            if (pending == kOutTile) {
                Flush(outLocal, flushBase, pending);
                flushBase = row + 1;
                pending = 0;
            }
        }
        if (pending > 0) {
            Flush(outLocal, flushBase, pending);
        }
        colQue_.FreeTensor(colLocal);
        outQue_.FreeTensor(outLocal);
    }

private:
    /**
     * B 每行等长（长度 L = bUniformLen_）时的解析计数。
     *
     *   rowProducts[r] = (aRowPtr[r+1] − aRowPtr[r]) · L
     *
     * 无需读 aColIdx 或 bRowPtr，每行只需一次 aRowPtr 读。
     * 越界列索引按 L 计入（上界），与下游契约相容。
     */
    __aicore__ inline void CountRowProductsUniform()
    {
        LocalTensor<int64_t> outLocal = outQue_.AllocTensor<int64_t>();
        const int64_t L = static_cast<int64_t>(bUniformLen_);
        uint32_t pending = 0;
        uint32_t flushBase = rowStart_;
        int32_t prev = aRowPtrGm_.GetValue(rowStart_);
        for (uint32_t row = rowStart_; row < rowEnd_; row++) {
            const int32_t next = aRowPtrGm_.GetValue(row + 1);
            const int32_t len = next - prev;
            prev = next;
            outLocal.SetValue(pending, (len > 0) ? static_cast<int64_t>(len) * L : 0);
            pending++;
            if (pending == kOutTile) {
                Flush(outLocal, flushBase, pending);
                flushBase = row + 1;
                pending = 0;
            }
        }
        if (pending > 0) {
            Flush(outLocal, flushBase, pending);
        }
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void Flush(const LocalTensor<int64_t> &outLocal,
                                 uint32_t base, uint32_t count)
    {
        // 标量写 UB 后需等待再由 MTE3 搬出。
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(rowProductsGm_[base], outLocal, cp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void FlushAp(const LocalTensor<int32_t> &apLocal,
                                   const LocalTensor<int32_t> &headLocal,
                                   uint32_t base, uint32_t count)
    {
        // 描述符可能由向量或标量写入，两条都得在 MTE3 读之前落地。
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(bApStrideGm_[base], apLocal, cp);
        DataCopyPad(bHeadGm_[base], headLocal, cp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> colQue_;
    TQue<TPosition::VECOUT, 1> outQue_;
    TQue<TPosition::VECOUT, 1> apQue_;
    TQue<TPosition::VECOUT, 1> headQue_;
    TQue<TPosition::VECIN, 1> syncQue_;
    // 向量瓦片工作区（列索引瓦片 + 中间量 + 行不变索引表）。
    TBuf<TPosition::VECCALC> vecBuf_;

    GlobalTensor<int32_t> aRowPtrGm_;
    GlobalTensor<int32_t> aColIdxGm_;
    GlobalTensor<int32_t> bRowPtrGm_;
    GlobalTensor<int32_t> bColIdxGm_;
    GlobalTensor<int64_t> rowProductsGm_;
    GlobalTensor<int32_t> bApStrideGm_;
    GlobalTensor<int32_t> bHeadGm_;
    GlobalTensor<int32_t> bStatGm_;
    GlobalTensor<int32_t> syncGm_;

    uint32_t M_ = 0;
    uint32_t K_ = 0;
    uint32_t nnzA_ = 0;
    uint32_t nnzB_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t kStart_ = 0;
    uint32_t kEnd_ = 0;
    uint32_t blockDim_ = 1;
    // B 行长统计。空 k 区间的核保持中性初值，跨核归约时自动被忽略。
    int32_t minLen_ = 0x7FFFFFFF;
    int32_t maxLen_ = -0x7FFFFFFF;
    // > 0 表示 B 每行等长且长度为该值；0 表示不等长，走逐段路径。
    int32_t bUniformLen_ = 0;
    // B「带状前缀」参考三元组与首个违规行号。
    int32_t bandL_ = 0;
    int32_t bandS_ = 0;
    int32_t bandH_ = 0;
    uint32_t bandBad_ = 0x7FFFFFFFu;
    bool bandRefOk_ = false;
    // 行不变索引表当前对应的行长 L；0 表示尚未建表。
    int32_t tblL_ = 0;
    // jBase[j]=j 与 L 无关，只建一次。
    bool rampOk_ = false;
};

}  // namespace

__global__ __aicore__ void spgemm_arch22_count(
    GM_ADDR aRowPtr, GM_ADDR aColIdx, GM_ADDR bRowPtr, GM_ADDR bColIdx,
    GM_ADDR buffer1, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpgemmArch22CountKernel op;
    op.Init(aRowPtr, aColIdx, bRowPtr, bColIdx, buffer1, tilingGm);
    op.Process();
}

void spgemm_arch22_count_launch(
    void *aRowPtr, void *aColIdx, void *bRowPtr, void *bColIdx,
    void *buffer1, void *tiling, uint32_t blockDim, void *stream)
{
    spgemm_arch22_count<<<blockDim, nullptr, stream>>>(
        (GM_ADDR)aRowPtr, (GM_ADDR)aColIdx, (GM_ADDR)bRowPtr, (GM_ADDR)bColIdx,
        (GM_ADDR)buffer1, (GM_ADDR)tiling);
}
