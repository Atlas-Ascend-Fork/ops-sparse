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
 * \file spgemm_merge_row.h
 * \brief SpGEMM arch22 归并的单行实现（自 spgemm_merge.h 切出）。
 *
 * 承载：
 *   - SpgemmMergeCountRowImpl / SpgemmMergeCountRow        计数阶段的单行归并
 *   - SpgemmMergeDisjointOrder / SpgemmMergeEmitDisjoint   段间不交叠冷块
 *   - SpgemmMergeFusedRowGeneral                           通用 k 路归并兜底
 *   - SpgemmMergeFusedRowImpl                              DA_FIXED 特化的行入口
 *   - spgemm_merge_row_fastpaths.h                        轮转、不交叠及 Complex64 特化
 *
 * 计数版与融合版的推进规则必须保持同步——改一处必须同步改另一处。
 */

// 本文件是 spgemm_merge.h 的内部片段，只允许由 spgemm_merge.h 在原定义位置
// #include，不是独立可用的头文件。
#ifndef SPGEMM_ARCH22_MERGE_H
#error "spgemm_merge_{emit,row}.h must only be included from spgemm_merge.h"
#endif

#ifndef SPGEMM_ARCH22_MERGE_ROW_H
#define SPGEMM_ARCH22_MERGE_ROW_H


struct SpgemmMinScanResult {
    int32_t minCol;
    uint64_t hitMask;
};

template <uint32_t NW>
__aicore__ inline SpgemmMinScanResult SpgemmMergeFindMinCol(
    const int32_t (&head)[NW], uint32_t da)
{
    constexpr int32_t kExhausted = 0x7FFFFFFF;
    int32_t minCol = kExhausted; uint64_t hitMask = 0;
    for (uint32_t s = 0; s < da; s++) {
        const int32_t h = head[s];
        if (h < minCol) {
            minCol = h; hitMask = static_cast<uint64_t>(1) << s;
        } else if (h == minCol) {
            hitMask |= static_cast<uint64_t>(1) << s;
        }
    }
    return SpgemmMinScanResult{minCol, hitMask};
}

template <uint32_t NW>
[[gnu::always_inline]] __aicore__ inline SpgemmMinScanResult SpgemmMergeFindMinColTwoPass(
    const int32_t (&head)[NW], uint32_t da)
{
    constexpr int32_t kExhausted = 0x7FFFFFFF;
    int32_t minCol = kExhausted;
    for (uint32_t s = 0; s < da; s++) {
        minCol = (head[s] < minCol) ? head[s] : minCol;
    }
    // Separating the reductions avoids resetting the mask when the minimum changes.
    uint64_t hitMask = 0;
    for (uint32_t s = 0; s < da; s++) {
        hitMask |= static_cast<uint64_t>(head[s] == minCol) << s;
    }
    return SpgemmMinScanResult{minCol, hitMask};
}


template <uint32_t DA_FIXED, bool COALESCE_B = false> __aicore__ inline int32_t SpgemmMergeCountRowImpl(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, SpgemmSegments &seg, uint32_t daRun)
{
    constexpr int32_t kExhausted = 0x7FFFFFFF; constexpr bool kFixed = (DA_FIXED != 0); const uint32_t da = kFixed ? DA_FIXED : daRun;

    int32_t nnz = 0; int32_t head[kFixed ? DA_FIXED : SPGEMM_ARCH22_MAX_WAYS];
    // 特化路径的段首取建段时顺带取出的 seg.first[]，不再读 GM。
    if constexpr (kFixed && DA_FIXED <= SPGEMM_ARCH22_FIRST_WAYS) {
        for (uint32_t s = 0; s < da; s++) { head[s] = seg.first[s]; }
    } else {
        for (uint32_t s = 0; s < da; s++) {
            head[s] = (seg.cur[s] < seg.end[s]) ? bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s])) : kExhausted;
        }
    }

    // 等步长轮转快路径（计数版）：满足时输出数恒为 segLen * da，直接返回。
    if constexpr (kFixed) {
        if (SpgemmMergeIsRotational<DA_FIXED>(seg, head)) {
            return static_cast<int32_t>(static_cast<uint32_t>(seg.segLenAll) * DA_FIXED);
        }
    }

    while (true) {
        const SpgemmMinScanResult scan = SpgemmMergeFindMinCol(head, da);
        if (scan.minCol == kExhausted) { break; }
        uint64_t hitMask = scan.hitMask;
        while (hitMask != 0) {
            const uint32_t s = static_cast<uint32_t>(__builtin_ctzll(hitMask)); hitMask &= hitMask - 1;
            if constexpr (COALESCE_B) {
                do {
                    seg.cur[s]++;
                    head[s] = (seg.cur[s] < seg.end[s]) ? bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s])) : kExhausted;
                } while (head[s] == scan.minCol);
            } else {
                seg.cur[s]++;
                head[s] = (seg.cur[s] < seg.end[s]) ? bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s])) : kExhausted;
            }
        }
        nnz++;
    }
    return nnz;
}

template <bool COALESCE_B = false>
__aicore__ inline int32_t SpgemmMergeCountRow(const AscendC::GlobalTensor<int32_t> &bColIdxGm, SpgemmSegments &seg)
{
    const uint32_t da = seg.count;
    if (da == 0) {
        return 0;
    }

    if (da == 1) {
        if constexpr (COALESCE_B) {
            return SpgemmMergeCountRowImpl<1, true>(bColIdxGm, seg, da);
        } else {
            return seg.end[0] - seg.cur[0];
        }
    }

    // 按段数编译期特化 {2,3,4,6,7,8}，其余走通用路径。
    switch (da) {
        case 2:  return SpgemmMergeCountRowImpl<2, COALESCE_B>(bColIdxGm, seg, da); case 3:  return SpgemmMergeCountRowImpl<3, COALESCE_B>(bColIdxGm, seg, da);
        case 4:  return SpgemmMergeCountRowImpl<4, COALESCE_B>(bColIdxGm, seg, da); case 6:  return SpgemmMergeCountRowImpl<6, COALESCE_B>(bColIdxGm, seg, da);
        case 7:  return SpgemmMergeCountRowImpl<7, COALESCE_B>(bColIdxGm, seg, da); case 8:  return SpgemmMergeCountRowImpl<8, COALESCE_B>(bColIdxGm, seg, da);
        default: return SpgemmMergeCountRowImpl<0, COALESCE_B>(bColIdxGm, seg, da);
    }
}

/**
 * 不交叠快路径的判定：算出段首序 ord[]，并同时判定
 *   (1) 相邻段区间两两不交叠  (2) 输出总数不超 cap
 *
 * @param rank 各段段首的秩
 * @param head 各段段首列索引
 * @param ord  输出：按段首序排列的段下标
 */
template <uint32_t DA_FIXED, typename T> [[gnu::always_inline]] __aicore__ inline bool SpgemmMergeDisjointOrder(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED], const int32_t (&head)[DA_FIXED],
    uint32_t cap, uint32_t (&ord)[DA_FIXED])
{
    int32_t segLast[DA_FIXED];
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        segLast[s] = bColIdxGm.GetValue(static_cast<uint64_t>(seg.end[s] - 1));
    }
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        ord[rank[s]] = s;
    }
    bool disjoint = true;
    for (uint32_t i = 0; i + 1 < DA_FIXED; i++) {
        if (segLast[ord[i]] >= head[ord[i + 1]]) {
            disjoint = false;
        }
    }
    if (!disjoint) {
        return false;
    }
    uint32_t total = 0;
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        total += static_cast<uint32_t>(seg.end[s] - seg.cur[s]);
    }
    return total <= cap;
}

template <typename T, bool UNIT_ALPHA = false> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitOne(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops, float aRe, float aIm, int32_t p, int32_t col,
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal,
    int32_t &outCount)
{
    float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f;
    ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bRe, bIm);
    ops.Accumulate(aRe, aIm, bRe, bIm, sumRe, sumIm);
    if constexpr (!UNIT_ALPHA) { ops.ApplyAlpha(sumRe, sumIm); }
    outCol.SetValue(static_cast<uint32_t>(outCount), col);
    ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumRe, sumIm);
    outCount++;
}

template <typename T, bool UNIT_ALPHA = false> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitCoalescedRun(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    float aRe, float aIm, int32_t end, int32_t &p, int32_t &currentCol, const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, int32_t &outCount)
{
    const int32_t col = currentCol; float sumRe = 0.0f; float sumIm = 0.0f;
    float bR0 = 0.0f; float bI0 = 0.0f; float bR1 = 0.0f; float bI1 = 0.0f;
    ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bR0, bI0); ops.LoadRaw(bValGm, static_cast<uint32_t>(p + 1), bR1, bI1);
    ops.Accumulate(aRe, aIm, bR0, bI0, sumRe, sumIm); ops.Accumulate(aRe, aIm, bR1, bI1, sumRe, sumIm);
    p += 2; if (p < end) { currentCol = bColIdxGm.GetValue(static_cast<uint64_t>(p)); }
    while (p < end && currentCol == col) {
        float bRe = 0.0f; float bIm = 0.0f; ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bRe, bIm);
        ops.Accumulate(aRe, aIm, bRe, bIm, sumRe, sumIm); p++;
        if (p < end) { currentCol = bColIdxGm.GetValue(static_cast<uint64_t>(p)); }
    }
    if constexpr (!UNIT_ALPHA) { ops.ApplyAlpha(sumRe, sumIm); } outCol.SetValue(static_cast<uint32_t>(outCount), col);
    ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumRe, sumIm); outCount++;
}

template <typename T, bool UNIT_ALPHA = false> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitCoalescedSegment(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    const SpgemmSegments &seg, uint32_t s, int32_t head, const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, int32_t &outCount)
{
    const int32_t end = seg.end[s]; int32_t p = seg.cur[s]; int32_t currentCol = head;
    while (p + 1 < end) {
        const int32_t col0 = currentCol; const int32_t col1 = bColIdxGm.GetValue(static_cast<uint64_t>(p + 1));
        if (col1 == col0) { SpgemmMergeEmitCoalescedRun<T, UNIT_ALPHA>(bColIdxGm, bValGm, ops, seg.valRe[s], seg.valIm[s], end, p, currentCol, outCol, outVal, wideVal, outCount); continue; }
        const bool hasThird = (p + 2 < end); const int32_t col2 = hasThird ? bColIdxGm.GetValue(static_cast<uint64_t>(p + 2)) : col1;
        if (!hasThird || col2 != col1) {
            float sumR0 = 0.0f; float sumI0 = 0.0f; float sumR1 = 0.0f; float sumI1 = 0.0f;
            float bR0 = 0.0f; float bI0 = 0.0f; float bR1 = 0.0f; float bI1 = 0.0f;
            ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bR0, bI0);
            ops.LoadRaw(bValGm, static_cast<uint32_t>(p + 1), bR1, bI1);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bR0, bI0, sumR0, sumI0);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bR1, bI1, sumR1, sumI1);
            if constexpr (!UNIT_ALPHA) { ops.ApplyAlpha(sumR0, sumI0); ops.ApplyAlpha(sumR1, sumI1); }
            outCol.SetValue(static_cast<uint32_t>(outCount), col0);
            outCol.SetValue(static_cast<uint32_t>(outCount + 1), col1);
            ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumR0, sumI0);
            ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount + 1), sumR1, sumI1);
            outCount += 2; p += 2; currentCol = col2; continue;
        }
        float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f;
        ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bRe, bIm);
        ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm);
        if constexpr (!UNIT_ALPHA) { ops.ApplyAlpha(sumRe, sumIm); }
        outCol.SetValue(static_cast<uint32_t>(outCount), col0);
        ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumRe, sumIm);
        outCount++; p++; currentCol = col1;
    }
    if (p < end) { SpgemmMergeEmitOne<T, UNIT_ALPHA>(bValGm, ops, seg.valRe[s], seg.valIm[s], p, currentCol, outCol, outVal, wideVal, outCount); }
}

/** 不交叠快路径的发射：按段首序逐段整段写出，段内保持 B 的列序，2 路 ILP 展开。 */
template <uint32_t DA_FIXED, bool COALESCE_B, typename T, bool UNIT_ALPHA = false> [[gnu::always_inline]] __aicore__ inline void SpgemmMergeEmitDisjoint(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    const SpgemmSegments &seg, const uint32_t (&ord)[DA_FIXED], const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, int32_t &outCount)
{
    for (uint32_t i = 0; i < DA_FIXED; i++) {
        const uint32_t s = ord[i];
        if constexpr (COALESCE_B) {
            SpgemmMergeEmitCoalescedSegment<T, UNIT_ALPHA>(bColIdxGm, bValGm, ops, seg, s, head[s], outCol, outVal, wideVal, outCount);
            continue;
        }
        const int32_t end = seg.end[s]; int32_t p = seg.cur[s];
        for (; p + 1 < end; p += 2) {
            float sumR0 = 0.0f, sumI0 = 0.0f; float sumR1 = 0.0f, sumI1 = 0.0f; float bR0 = 0.0f, bI0 = 0.0f; float bR1 = 0.0f, bI1 = 0.0f;
            ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bR0, bI0); ops.LoadRaw(bValGm, static_cast<uint32_t>(p + 1), bR1, bI1);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bR0, bI0, sumR0, sumI0); ops.ApplyAlpha(sumR0, sumI0);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bR1, bI1, sumR1, sumI1); ops.ApplyAlpha(sumR1, sumI1);
            outCol.SetValue(static_cast<uint32_t>(outCount), bColIdxGm.GetValue(static_cast<uint64_t>(p)));
            outCol.SetValue(static_cast<uint32_t>(outCount + 1), bColIdxGm.GetValue(static_cast<uint64_t>(p + 1)));
            ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumR0, sumI0);
            ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount + 1), sumR1, sumI1); outCount += 2;
        }
        if (p < end) {
            float sumRe = 0.0f; float sumIm = 0.0f; float bRe = 0.0f; float bIm = 0.0f; ops.LoadRaw(bValGm, static_cast<uint32_t>(p), bRe, bIm);
            ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm); ops.ApplyAlpha(sumRe, sumIm);
            outCol.SetValue(static_cast<uint32_t>(outCount), bColIdxGm.GetValue(static_cast<uint64_t>(p)));
            ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumRe, sumIm); outCount++;
        }
    }
}

/**
 * 通用 k 路归并主循环（最终回退路径）。
 *
 * @param head     各段当前段首列索引
 * @param outCount 进入时已写出的条目数
 * @return 行内去重后的列数；-1 表示超出容量
 */
template <uint32_t DA_FIXED, bool COALESCE_B, typename T> __aicore__ inline int32_t SpgemmMergeFusedRowGeneral(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, uint32_t da, int32_t (&head)[DA_FIXED != 0 ? DA_FIXED : SPGEMM_ARCH22_MAX_WAYS],
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, uint32_t cap, int32_t outCount)
{
    constexpr int32_t kExhausted = 0x7FFFFFFF;
    // 补齐 seg.end[]（带路径建段省掉了它）。
    if (seg.segLenAll > 0) {
        for (uint32_t s = 0; s < da; s++) {
            seg.end[s] = seg.cur[s] + seg.segLenAll;
        }
    }

    while (true) {
        SpgemmMinScanResult scan{};
        if constexpr (COALESCE_B && DA_FIXED != 0 && AscendC::IsSameType<T, float>::value) {
            scan = SpgemmMergeFindMinColTwoPass<DA_FIXED>(head, da);
        } else {
            scan = SpgemmMergeFindMinCol(head, da);
        }
        if (scan.minCol == kExhausted) {
            break;
        }
        if (static_cast<uint32_t>(outCount) >= cap) {
            return -1;
        }
        float sumRe = 0.0f; float sumIm = 0.0f;
        uint64_t hitMask = scan.hitMask;
        while (hitMask != 0) {
            const uint32_t s = static_cast<uint32_t>(__builtin_ctzll(hitMask)); hitMask &= hitMask - 1; float bRe = 0.0f; float bIm = 0.0f;
            if constexpr (COALESCE_B) {
                do {
                    ops.LoadRaw(bValGm, static_cast<uint32_t>(seg.cur[s]), bRe, bIm); ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm);
                    seg.cur[s]++; head[s] = (seg.cur[s] < seg.end[s]) ? bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s])) : kExhausted;
                } while (head[s] == scan.minCol);
            } else {
                ops.LoadRaw(bValGm, static_cast<uint32_t>(seg.cur[s]), bRe, bIm); ops.Accumulate(seg.valRe[s], seg.valIm[s], bRe, bIm, sumRe, sumIm);
                seg.cur[s]++; head[s] = (seg.cur[s] < seg.end[s]) ? bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s])) : kExhausted;
            }
        }
        ops.ApplyAlpha(sumRe, sumIm); outCol.SetValue(static_cast<uint32_t>(outCount), scan.minCol);
        ops.StoreValDispatch(outVal, wideVal, static_cast<uint32_t>(outCount), sumRe, sumIm); outCount++;
    }
    return outCount;
}


#include "spgemm_merge_row_fastpaths.h"

/**
 * 单行融合归并（Variant B）：在符号阶段的同一趟归并里同时产出列索引与数值。
 *
 * @tparam DA_FIXED 段数的编译期值；0 表示运行时段数（通用回退）
 * @param cap       outCol/outVal 的容量
 * @param aBase     本行 da 个段的 A 值在 aValGm 中的起始元素下标（无段被跳过时非负，否则 −1）
 * @return 行内去重后的列数；-1 表示超出容量
 */
template <uint32_t DA_FIXED, bool COALESCE_B, typename T, typename VEC>
__aicore__ inline int32_t SpgemmMergeFusedRowImpl(const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, uint32_t daRun,
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, uint32_t cap, const VEC &vec, uint32_t vecSlotOff,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase)
{
    constexpr bool kFixed = (DA_FIXED != 0); const uint32_t da = kFixed ? DA_FIXED : daRun;

    int32_t outCount = 0; int32_t head[kFixed ? DA_FIXED : SPGEMM_ARCH22_MAX_WAYS];
    // 段首载入（seg.first[] 直取或 GM 读）。
    SpgemmMergeRowLoadHeads<DA_FIXED, kFixed>(bColIdxGm, seg, da, head);

    if constexpr (kFixed) {
        // 秩计算与轮转判定。
        uint32_t rank[DA_FIXED];
        const SpgemmRotationalGateOut gate =
            SpgemmMergeRowRotationalGate<DA_FIXED, T>(seg, head, cap, rank);
        const int32_t hdStep = gate.hdStep; const int32_t stride = gate.stride;
        const uint32_t segLen = gate.segLen; const bool rotational = gate.rotational;
        if (rotational) {
            return SpgemmMergeEmitRotational<DA_FIXED, T, VEC>(bValGm, ops, seg, da, rank, head,
                    outCol, outVal, wideVal, cap, vec, vecSlotOff, aValGm, aBase, hdStep, stride,
                    segLen, gate.c64VectorAllowed);
        }

        // 不交叠快路径（按模板属性选择性保留）。
        if constexpr (COALESCE_B && SpgemmValueOps<T>::kIsComplex) {
            // Choose once per row so the coalesced emitter has no alpha branch per output.
            const int32_t coldCount = SpgemmMergeFusedRowColdTailC64Dispatch<DA_FIXED, T>(
                bColIdxGm, bValGm, ops, seg, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase);
            if (coldCount >= 0) { return coldCount; }
        } else {
            if (SpgemmMergeFusedRowColdTail<DA_FIXED, COALESCE_B, T>(bColIdxGm, bValGm, ops, seg, da, rank, head,
                    outCol, outVal, wideVal, cap, aValGm, aBase, outCount)) {
                return outCount;
            }
        }
    }

    // 通用建段已载入 A values；固定段数的 Band 路径已在上方尾段补齐。
    if constexpr (COALESCE_B && SpgemmValueOps<T>::kIsComplex && DA_FIXED == SPGEMM_ARCH22_FIRST_WAYS) {
        // Select once per row so unit alpha generates no speculative complex multiplication.
        if (ops.alphaRe == 1.0f && ops.alphaIm == 0.0f) {
            return SpgemmMergeFusedRowGeneralC64<DA_FIXED, true, T>(
                bColIdxGm, bValGm, ops, seg, outCol, outVal, cap);
        }
        return SpgemmMergeFusedRowGeneralC64<DA_FIXED, false, T>(
            bColIdxGm, bValGm, ops, seg, outCol, outVal, cap);
    }
    return SpgemmMergeFusedRowGeneral<DA_FIXED, COALESCE_B, T>(bColIdxGm, bValGm, ops, seg, da, head, outCol, outVal, wideVal, cap, outCount);
}

#endif  // SPGEMM_ARCH22_MERGE_ROW_H
