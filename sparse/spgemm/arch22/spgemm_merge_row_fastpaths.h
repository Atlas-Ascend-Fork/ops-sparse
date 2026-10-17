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
 * \file spgemm_merge_row_fastpaths.h
 * \brief SpGEMM arch22 单行归并的轮转、不交叠及 Complex64 特化路径。
 */

// 内部实现片段，依赖 spgemm_merge_row.h 中先定义的通用归并与发射函数。
#ifndef SPGEMM_ARCH22_MERGE_ROW_H
#error "spgemm_merge_row_fastpaths.h must only be included from spgemm_merge_row.h"
#endif

#ifndef SPGEMM_ARCH22_MERGE_ROW_FASTPATHS_H
#define SPGEMM_ARCH22_MERGE_ROW_FASTPATHS_H

/**
 * 每行段首载入。必须 always_inline。
 */
template <uint32_t DA_FIXED, bool kFixed, uint32_t NW>
[[gnu::always_inline]] __aicore__ inline void SpgemmMergeRowLoadHeads(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const SpgemmSegments &seg, uint32_t da,
    int32_t (&head)[NW])
{
    if constexpr (kFixed && DA_FIXED <= SPGEMM_ARCH22_FIRST_WAYS) {
    for (uint32_t s = 0; s < da; s++) { head[s] = seg.first[s]; }
    } else {
    for (uint32_t s = 0; s < da; s++) {
        head[s] = bColIdxGm.GetValue(static_cast<uint64_t>(seg.cur[s]));
    }
    }
}


/** SpgemmMergeRowRotationalGate 的回传包（按值返回，避免出参落栈）。 */
struct SpgemmRotationalGateOut {
    int32_t hdStep;
    int32_t stride;
    uint32_t segLen;
    bool rotational;
    bool c64VectorAllowed;
};

/**
 * 段首秩计算（升序快路径 + O(da²) 插入定序）。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED>
[[gnu::always_inline]] __aicore__ inline void SpgemmMergeRowRank(
    const int32_t (&head)[DA_FIXED], uint32_t (&rank)[DA_FIXED], uint32_t &eqBad,
    int32_t &minFirst, int32_t &maxFirst)
{
    uint32_t asc = 0u;
    if constexpr (DA_FIXED >= 6u) {
        asc = 1u;
        for (uint32_t s = 1; s < DA_FIXED; s++) {
            asc &= (head[s - 1] < head[s]) ? 1u : 0u;
        }
    }
    if (asc != 0u) {
        for (uint32_t s = 0; s < DA_FIXED; s++) {
            rank[s] = s;
        }
        maxFirst = head[DA_FIXED - 1];
    } else {
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        rank[s] = 0;
    }
    for (uint32_t s = 1; s < DA_FIXED; s++) {
        const int32_t hs = head[s];
        for (uint32_t u = 0; u < s; u++) {
            const int32_t hu = head[u]; rank[s] += (hu <= hs) ? 1u : 0u; rank[u] += (hs < hu) ? 1u : 0u; eqBad |= (hu == hs) ? 1u : 0u;
        }
        if (hs < minFirst) {
            minFirst = hs;
        }
        if (hs > maxFirst) {
            maxFirst = hs;
        }
    }
    }
}


/**
 * 每行的秩计算 + 轮转判定。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T>
[[gnu::always_inline]] __aicore__ inline SpgemmRotationalGateOut SpgemmMergeRowRotationalGate(
    const SpgemmSegments &seg, const int32_t (&head)[DA_FIXED], uint32_t cap,
    uint32_t (&rank)[DA_FIXED])
{
    uint32_t eqBad = 0; int32_t minFirst = head[0]; int32_t maxFirst = head[0];
    SpgemmMergeRowRank<DA_FIXED>(head, rank, eqBad, minFirst, maxFirst);
    // 段首等差判据（只在复数实例计算，实数实例由各发射块入口内现算）。
    constexpr bool kNeedAp = SpgemmValueOps<T>::kIsComplex && DA_FIXED >= 8u;
    int32_t hdStep = 0;
    bool c64VectorAllowed = true;
    if constexpr (kNeedAp) {
        hdStep = head[1] - head[0]; uint32_t apBad = (hdStep > 0) ? 0u : 1u;
        for (uint32_t s = 2; s < DA_FIXED; s++) {
            apBad |= static_cast<uint32_t>(head[s] - head[s - 1] - hdStep);
        }
        c64VectorAllowed = (apBad == 0u);
    }
    // 等步长轮转判定。
    int32_t stride = 0;
    uint32_t segLen = static_cast<uint32_t>(seg.segLenAll); bool rotational = (seg.segLenAll >= 2);
    if (rotational) {
        stride = seg.apStride;
        rotational = (stride > 0) && (eqBad == 0) && (maxFirst - minFirst < stride) && (segLen * DA_FIXED <= cap);
    }
    return SpgemmRotationalGateOut{hdStep, stride, segLen, rotational, c64VectorAllowed};
}


/**
 * 轮转快路径的尾段发射（c64 向量块 + A 值补齐 + 标量发射）。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC>
[[gnu::always_inline]] __aicore__ inline int32_t SpgemmMergeEmitRotationalTail(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    SpgemmSegments &seg, uint32_t da, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal,
    uint32_t cap, const VEC &vec, uint32_t vecSlotOff,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase, int32_t hdStep, int32_t stride,
    uint32_t segLen, uint32_t total, bool c64VectorAllowed)
{
    int32_t outCount = 0;
    // complex64 向量发射块。
    if (c64VectorAllowed && SpgemmMergeEmitVecC64<DA_FIXED, T, VEC>(bValGm, ops, seg, outCol, outVal, vec, vecSlotOff, aValGm,
            aBase, head, hdStep, stride, segLen, total)) {
        return static_cast<int32_t>(total);
    }
    // 补齐 A 值（向量块未截走时，轮转标量发射之前）。
    if (aBase >= 0) {
        for (uint32_t s = 0; s < da; s++) {
            ops.LoadRaw(aValGm, static_cast<uint32_t>(aBase) + s, seg.valRe[s], seg.valIm[s]);
        }
    }
    // 标量发射：按元素宽度选段主序或层主序。
    constexpr bool kVecEmitInstantiated = VEC::kUsable && !SpgemmValueTraits<T>::kIsComplex && DA_FIXED >= 8u && (DA_FIXED & 7u) == 0u;
    constexpr bool kSegMajor = !SpgemmValueTraits<T>::kIsComplex && (!AscendC::IsSameType<T, bfloat16_t>::value || !kVecEmitInstantiated);
    if constexpr (kSegMajor) {
    for (uint32_t s = 0; s < DA_FIXED; s++) {
        SpgemmMergeEmitSegMajorOne<DA_FIXED, T>(bValGm, ops, seg, rank, head, s, outCol, outVal, wideVal, stride, segLen);
    }
    } else {
        SpgemmMergeEmitLayerMajor<DA_FIXED, T>(bValGm, ops, seg, rank, head, outCol, outVal, wideVal, stride, segLen);
    }
    outCount = static_cast<int32_t>(total);
    return outCount;
}


/**
 * 轮转快路径的发射段整体。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, typename T, typename VEC>
[[gnu::always_inline]] __aicore__ inline int32_t SpgemmMergeEmitRotational(
    const AscendC::GlobalTensor<T> &bValGm, const SpgemmValueOps<T> &ops,
    SpgemmSegments &seg, uint32_t da, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal,
    uint32_t cap, const VEC &vec, uint32_t vecSlotOff,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase, int32_t hdStep, int32_t stride,
    uint32_t segLen, bool c64VectorAllowed)
{
        const uint32_t total = segLen * DA_FIXED;
        // da 为 8 倍数的实型向量发射块。
        if (SpgemmMergeEmitVec8<DA_FIXED, T, VEC>(bValGm, ops, seg, outCol, outVal, wideVal, vec, vecSlotOff,
                aValGm, aBase, head, stride, segLen, total)) {
            return static_cast<int32_t>(total);
        }
        // da == 7 的向量化发射。
        if constexpr (VEC::kUsable && !SpgemmValueOps<T>::kIsComplex &&
                      DA_FIXED == 7u) {
            const uint32_t pad = vecSlotOff & 7u;
            if (vec.enabled && total <= kSpgemmVecEmitCap &&
                ((pad + total) <= 64u) && ((cap + pad) >= 64u) && aBase >= 0) {
                // 逐段等距 + 等差判据（在发射块入口内现算）。
                const uint32_t minCur = static_cast<uint32_t>(seg.cur[0]); uint32_t spanBad = 0; const int32_t hdStep = head[1] - head[0];
                uint32_t prev = minCur; spanBad = (hdStep > 0) ? 0u : 1u;
                for (uint32_t s = 1; s < DA_FIXED; s++) {
                    const uint32_t c = static_cast<uint32_t>(seg.cur[s]); spanBad |= (c - prev - segLen);
                    spanBad |= static_cast<uint32_t>(head[s] - head[s - 1] - hdStep); prev = c;
                }
                if (spanBad == 0u) {
                    SpgemmMergeEmitVec7Body<DA_FIXED, T, VEC>(bValGm, ops, vec, vecSlotOff, aValGm, aBase, head, hdStep,
                        stride, segLen, total, pad, minCur);
                    return static_cast<int32_t>(total);
                }
            }
        }
        // 轮转尾段。
        return SpgemmMergeEmitRotationalTail<DA_FIXED, T, VEC>(bValGm, ops, seg, da, rank, head,
                outCol, outVal, wideVal, cap, vec, vecSlotOff, aValGm, aBase, hdStep, stride,
                segLen, total, c64VectorAllowed);
}


/**
 * 轮转全部未命中后的冷尾段（不交叠 + seg.end 补齐 + A 值补齐）。
 * 必须 always_inline。
 */
template <uint32_t DA_FIXED, bool COALESCE_B, typename T, bool UNIT_ALPHA = false>
[[gnu::always_inline]] __aicore__ inline bool SpgemmMergeFusedRowColdTail(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, uint32_t da,
    const uint32_t (&rank)[DA_FIXED], const int32_t (&head)[DA_FIXED],
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal,
    const AscendC::LocalTensor<float> &wideVal, uint32_t cap,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase, int32_t &outCount)
{
    // 补齐 A 值。
    if (aBase >= 0) {
        for (uint32_t s = 0; s < da; s++) {
            ops.LoadRaw(aValGm, static_cast<uint32_t>(aBase) + s, seg.valRe[s], seg.valIm[s]);
        }
    }
    // 补齐 seg.end[]（带路径建段省掉了它）。
    if (seg.segLenAll > 0) {
        for (uint32_t s = 0; s < DA_FIXED; s++) {
            seg.end[s] = seg.cur[s] + seg.segLenAll;
        }
    }
    constexpr bool kKeepDisjoint = COALESCE_B || SpgemmValueOps<T>::kIsComplex || DA_FIXED != 7u;
    if constexpr (kKeepDisjoint) {
        uint32_t ord[DA_FIXED];
        if (SpgemmMergeDisjointOrder<DA_FIXED, T>(bColIdxGm, seg, rank, head, cap, ord)) {
            SpgemmMergeEmitDisjoint<DA_FIXED, COALESCE_B, T, UNIT_ALPHA>(bColIdxGm, bValGm, ops, seg, ord, head, outCol, outVal, wideVal, outCount);
            return true;
        }
    }
    return false;
}

template <uint32_t DA_FIXED, typename T, bool UNIT_ALPHA>
[[gnu::noinline]] __aicore__ inline int32_t SpgemmMergeFusedRowColdTailC64(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, uint32_t cap,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase)
{
    static_assert(SpgemmValueOps<T>::kIsComplex, "");
    int32_t outCount = 0;
    if (SpgemmMergeFusedRowColdTail<DA_FIXED, true, T, UNIT_ALPHA>(
            bColIdxGm, bValGm, ops, seg, DA_FIXED, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase, outCount)) {
        return outCount;
    }
    return -1;
}

template <uint32_t DA_FIXED, typename T>
[[gnu::always_inline]] __aicore__ inline int32_t SpgemmMergeFusedRowColdTailC64Dispatch(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, const uint32_t (&rank)[DA_FIXED],
    const int32_t (&head)[DA_FIXED], const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, uint32_t cap,
    const AscendC::GlobalTensor<T> &aValGm, int32_t aBase)
{
    if constexpr (DA_FIXED == SPGEMM_ARCH22_FIRST_WAYS) {
        int32_t outCount = 0;
        if (ops.alphaRe == 1.0f && ops.alphaIm == 0.0f) {
            if (SpgemmMergeFusedRowColdTail<DA_FIXED, true, T, true>(
                    bColIdxGm, bValGm, ops, seg, DA_FIXED, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase, outCount)) {
                return outCount;
            }
        } else if (SpgemmMergeFusedRowColdTail<DA_FIXED, true, T, false>(
                bColIdxGm, bValGm, ops, seg, DA_FIXED, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase, outCount)) {
            return outCount;
        }
        return -1;
    }
    if (ops.alphaRe == 1.0f && ops.alphaIm == 0.0f) {
        return SpgemmMergeFusedRowColdTailC64<DA_FIXED, T, true>(
            bColIdxGm, bValGm, ops, seg, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase);
    }
    return SpgemmMergeFusedRowColdTailC64<DA_FIXED, T, false>(
        bColIdxGm, bValGm, ops, seg, rank, head, outCol, outVal, wideVal, cap, aValGm, aBase);
}

static_assert(SPGEMM_ARCH22_FIRST_WAYS == 8u, "C64 fixed merge requires eight segments");

struct SpgemmC64FixedHeads {
    int64_t h0; int64_t h1; int64_t h2; int64_t h3;
    int64_t h4; int64_t h5; int64_t h6; int64_t h7;
};

[[gnu::always_inline]] __aicore__ inline int64_t SpgemmMergeMinEight(const SpgemmC64FixedHeads &head)
{
    const int64_t p01 = (head.h0 < head.h1) ? head.h0 : head.h1;
    const int64_t p23 = (head.h2 < head.h3) ? head.h2 : head.h3;
    const int64_t p45 = (head.h4 < head.h5) ? head.h4 : head.h5;
    const int64_t p67 = (head.h6 < head.h7) ? head.h6 : head.h7;
    const int64_t p0123 = (p01 < p23) ? p01 : p23;
    const int64_t p4567 = (p45 < p67) ? p45 : p67;
    return (p0123 < p4567) ? p0123 : p4567;
}

template <uint32_t S, typename T>
[[gnu::always_inline]] __aicore__ inline void SpgemmMergeAccumulateFixedHead(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, int64_t minCol,
    int64_t &headSlot, float &sumRe, float &sumIm)
{
    if (headSlot != minCol) { return; }
    constexpr int32_t kExhausted = 0x7FFFFFFF;
    int32_t cur = seg.cur[S]; const int32_t end = seg.end[S]; float bRe = 0.0f; float bIm = 0.0f;
    do {
        ops.LoadRaw(bValGm, static_cast<uint32_t>(cur), bRe, bIm);
        ops.Accumulate(seg.valRe[S], seg.valIm[S], bRe, bIm, sumRe, sumIm);
        cur++;
        const int32_t nextHead = (cur < end) ? bColIdxGm.GetValue(static_cast<uint64_t>(cur)) : kExhausted;
        headSlot = static_cast<int64_t>(nextHead);
    } while (headSlot == minCol);
    seg.cur[S] = cur;
}

template <uint32_t DA_FIXED, bool UNIT_ALPHA, typename T>
[[gnu::noinline]] __aicore__ inline int32_t SpgemmMergeFusedRowGeneralC64(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg,
    const AscendC::LocalTensor<int32_t> &outCol, const AscendC::LocalTensor<T> &outVal, uint32_t cap)
{
    static_assert(DA_FIXED == SPGEMM_ARCH22_FIRST_WAYS, "");
    constexpr int64_t kExhausted = 0x7FFFFFFF;
    if (seg.segLenAll > 0) {
        for (uint32_t s = 0; s < DA_FIXED; s++) { seg.end[s] = seg.cur[s] + seg.segLenAll; }
    }
    SpgemmC64FixedHeads heads{static_cast<int64_t>(seg.first[0]), static_cast<int64_t>(seg.first[1]),
        static_cast<int64_t>(seg.first[2]), static_cast<int64_t>(seg.first[3]), static_cast<int64_t>(seg.first[4]),
        static_cast<int64_t>(seg.first[5]), static_cast<int64_t>(seg.first[6]), static_cast<int64_t>(seg.first[7])};
    int32_t outCount = 0;
    while (true) {
        const int64_t minCol = SpgemmMergeMinEight(heads);
        if (minCol == kExhausted) { return outCount; }
        if (static_cast<uint32_t>(outCount) >= cap) { return -1; }
        float sumRe = 0.0f; float sumIm = 0.0f;
        SpgemmMergeAccumulateFixedHead<0>(bColIdxGm, bValGm, ops, seg, minCol, heads.h0, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<1>(bColIdxGm, bValGm, ops, seg, minCol, heads.h1, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<2>(bColIdxGm, bValGm, ops, seg, minCol, heads.h2, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<3>(bColIdxGm, bValGm, ops, seg, minCol, heads.h3, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<4>(bColIdxGm, bValGm, ops, seg, minCol, heads.h4, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<5>(bColIdxGm, bValGm, ops, seg, minCol, heads.h5, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<6>(bColIdxGm, bValGm, ops, seg, minCol, heads.h6, sumRe, sumIm);
        SpgemmMergeAccumulateFixedHead<7>(bColIdxGm, bValGm, ops, seg, minCol, heads.h7, sumRe, sumIm);
        if constexpr (!UNIT_ALPHA) {
            float scaledRe = 0.0f;
            float scaledIm = 0.0f;
            ops.MulComplex(ops.alphaRe, ops.alphaIm, sumRe, sumIm, scaledRe, scaledIm);
            sumRe = scaledRe;
            sumIm = scaledIm;
        }
        outCol.SetValue(static_cast<uint32_t>(outCount), static_cast<int32_t>(minCol));
        ops.StoreVal(outVal, static_cast<uint32_t>(outCount), sumRe, sumIm);
        outCount++;
    }
}


#endif  // SPGEMM_ARCH22_MERGE_ROW_FASTPATHS_H
