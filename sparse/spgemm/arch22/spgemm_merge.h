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
 * \file spgemm_merge.h
 * \brief SpGEMM arch22 行内 k 路有序归并骨架，符号阶段与数值阶段共用。
 *
 * 两趟共用同一份骨架保证结构一致性：符号阶段数出的每行 nnz 与数值阶段写出的条目数
 * 必然相同。归并输出有序且去重，累加顺序按段序固定，同一输入 bit-wise 一致。
 * 数值抵消产生的显式零会被保留并计入 nnz(C)。
 */

#ifndef SPGEMM_ARCH22_MERGE_H
#define SPGEMM_ARCH22_MERGE_H

#include "kernel_operator.h"
#include "spgemm.h"
#include "spgemm_value.h"

/**
 * 段游标集合：描述行 i 的各个有序段在 B.colIndices 中的当前读位置与结束位置。
 * 段数 da = nnz(A[i])，上限 SPGEMM_ARCH22_MAX_WAYS；超过则由调用方走 T3 列分块路径。
 */
struct SpgemmSegments {
    // cur[s] / end[s]：段 s 在 B.colIndices 中的 [cur, end) 区间
    int32_t cur[SPGEMM_ARCH22_MAX_WAYS]; int32_t end[SPGEMM_ARCH22_MAX_WAYS];
    // 段 s 对应的 A 值（数值阶段用；符号阶段忽略）。complex64 用 re/im 双分量。
    float valRe[SPGEMM_ARCH22_MAX_WAYS]; float valIm[SPGEMM_ARCH22_MAX_WAYS];
    uint32_t count = 0;  // 有效段数 da
    // 各段共同的等差公差：> 0 表示全部一致，0 表示不一致或存在非等差段。
    // 用标量而非 per-seg 数组以避免发射循环期间的寄存器压力。
    int32_t apStride = 0;
    // 各段共同的段长：> 0 表示所有段等长，0 表示不等长。同样用标量。
    int32_t segLenAll = 0;
    // 各段首列索引，由建段循环从 bHead[k] 取得。段推进后须用 head[] 跟踪。
    // 长度取 FIRST_WAYS（= 8），da > 8 的通用路径从 GM 读段首。
    int32_t first[SPGEMM_ARCH22_FIRST_WAYS];
    // cur[] 的 min/max，仅 complex64 使用（非复数被 DCE）。不给默认初值以避免多余栈写。
    int32_t curMin; int32_t curMax;
};

/**
 * T3 列分块路径的 A 行列索引瓦片搬入：把 [ai, min(ai + COL_TILE, re)) 搬进 UB
 * 并完成 MTE2→S 同步，返回本瓦片的有效长度。
 * 符号阶段与数值阶段共用，各自调用点在其后才读 colLocal。
 *
 * @param aColIdxGm A 的 colIndices（GM）
 * @param ai        本瓦片在 A.colIndices 中的起始偏移
 * @param re        A 行的结束偏移（用于钳位最后一个瓦片）
 * @param colLocal  UB 上的瓦片缓冲（容量 >= COL_TILE）
 * @return 本瓦片的有效元素数，恒 <= SPGEMM_ARCH22_COL_TILE
 */
__aicore__ inline uint32_t SpgemmLoadAColTile(const AscendC::GlobalTensor<int32_t> &aColIdxGm, uint32_t ai, uint32_t re,
    const AscendC::LocalTensor<int32_t> &colLocal)
{
    uint32_t len = re - ai;
    if (len > SPGEMM_ARCH22_COL_TILE) {
        len = SPGEMM_ARCH22_COL_TILE;
    }
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0}; AscendC::DataCopyPad(colLocal, aColIdxGm[ai], cp, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); return len;
}

/** 单行符号归并：只数出去重后的列数。与数值阶段推进规则完全相同。 */
/** 单行符号归并的模板实现体。@tparam DA_FIXED 段数编译期值；0 为通用回退。 */
/** 等步长轮转快路径判定（计数版），满足时输出 segLen * da，调用方直接返回。 */
template <uint32_t DA_FIXED> __aicore__ inline bool SpgemmMergeIsRotational(const SpgemmSegments &seg, const int32_t (&head)[DA_FIXED])
{
    // 段长全等已折成 seg.segLenAll（>= 2 表示各段等长且长度足够）。
    bool rotational = (seg.segLenAll >= 2);
    // 公差一致性已由建段循环合并算出。
    const int32_t stride = seg.apStride; rotational = rotational && (stride > 0);
    if (!rotational) {
        return false;
    }
    int32_t minFirst = head[0]; int32_t maxFirst = head[0];
    for (uint32_t s = 1; s < DA_FIXED; s++) {
        if (head[s] < minFirst) {
            minFirst = head[s];
        }
        if (head[s] > maxFirst) {
            maxFirst = head[s];
        }
    }
    // 段首两两不等判定（计数版 / 融合版必须一致）。
    uint32_t eqBad = 0;
    for (uint32_t s = 1; s < DA_FIXED; s++) {
        for (uint32_t u = 0; u < s; u++) {
            const uint32_t x = static_cast<uint32_t>(head[s]) ^ static_cast<uint32_t>(head[u]); eqBad |= (~(x | (0u - x))) >> 31;
        }
    }
    return (eqBad == 0) && (maxFirst - minFirst < stride);
}

/** da=7 向量发射专用的 markBuf_ 分区表（单位 = cap 个元素）。 */
constexpr uint32_t kSpgemmVecEmitCap = 256u;  // 向量发射暂存块容量

constexpr uint32_t kSpgemmVec7Mod = 7u;      // ×cap: mod7Bytes
constexpr uint32_t kSpgemmVec7Layer = 8u;    // ×cap: layer7Idx
constexpr uint32_t kSpgemmVec7LayerB = 9u;   // ×cap: layer7Bytes
constexpr uint32_t kSpgemmVec7Col = 10u;     // ×cap: 行结果列（int32，对齐暂存）
constexpr uint32_t kSpgemmVec7Val = 11u;     // ×cap: 行结果值（float，对齐暂存）
constexpr uint32_t kSpgemmVec7Shift = 12u;   // ×cap: 8×64 = 512 个元素（占 2·cap）
constexpr uint32_t kSpgemmVec7Base = 14u;    // ×cap: bOff[8] / bCol[8] / bA[8]
constexpr uint32_t kSpgemmVec7ModIdx = 15u;  // ×cap: mod7Idx = j % 7（纯下标）
// da=8 实数发射块的段号表 iPat8[j] = j % 8（纯下标，行不变）。
// 槽 16 是实数实例的第一个空闲块。
constexpr uint32_t kSpgemmVec8ModIdx = 16u;  // ×cap: iPat8 = j % 8（纯下标）
// da=8 窄类型（fp16/bf16）的 A 值搬运暂存。fp32 时不使用。
constexpr uint32_t kSpgemmVec8ARaw = 17u;    // ×cap: A 值窄类型搬运暂存

/**
 * 向量化发射的上下文：三块 32B 对齐的 UB 暂存 + 容量/开关。
 * enabled == false 时向量路径不进入。
 * 本结构只放整个 kernel 生命期不变的东西，逐行变化的量走函数参数。
 */
template <typename T>
struct SpgemmVecEmitCtx {
    AscendC::LocalTensor<int32_t> off;    // Gather 字节偏移表（int32 视图，供 Adds 倍增）
    AscendC::LocalTensor<uint32_t> offU;  // 同一块 UB 的 uint32 视图，供 Gather 取用
    AscendC::LocalTensor<float> tile;     // B 值的连续窗口（fp32；窄类型为 Cast 后的结果）
    AscendC::LocalTensor<float> aTile;    // aVal 按层平铺
    // 窄类型（fp16/bf16）的 B 值原始窗口。DataCopyPad 按存储类型搬，
    // 故先落在这里，再用 Cast 展宽到 tile；fp32/c64 下不使用。
    AscendC::LocalTensor<T> narrow;
    // 三张行不变的层号表（Init 时建一次，行循环内只读）：
    //   layerIdx[j]   = j / da            供 outCol 用
    //   layerBytes[j] = (j / da) · 4      供 off 用
    //   layerCol[]    = 纯暂存
    AscendC::LocalTensor<int32_t> layerIdx; AscendC::LocalTensor<int32_t> layerBytes; AscendC::LocalTensor<int32_t> layerCol;
    // 批缓冲的批基址（每批写一次，非 per-row 量）。
    // da=7 的 total 不是 8 的倍数，行基址一般不 32B 对齐，需从批基址算出对齐起址。
    AscendC::LocalTensor<int32_t> baseCol; AscendC::LocalTensor<float> baseDst;
    uint32_t cap = 0;                     // 上面各块的容量（元素数）
    bool enabled = false; static constexpr bool kUsable = true;
    // 向量发射路径特化的 da，在特化集 {2,3,4,6,7,8} 中恒为 8。
    static constexpr uint32_t kVecDa = 8;
};

/**
 * 不可向量化时的空上下文（da != 8 的实例按值传递，落到零指令）。
 */
struct SpgemmVecEmitNone {
    static constexpr bool kUsable = false;
};

/**
 * 向量化发射的值落点选择。
 * 窄类型写 fp32 暂存 wideVal（收窄由 flush 的批量 Cast 统一完成），fp32 直接写 outVal。
 */
template <typename T> __aicore__ inline const AscendC::LocalTensor<float> &SpgemmVecEmitDst(
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal)
{
    if constexpr (SpgemmValueOps<T>::kDeferCast) {
        return wideVal;
    } else {
        return outVal;
    }
}

/**
 * B 值窗口一次搬入（连续 total 个元素）。
 * 窄类型先按存储类型搬进 narrow，再 Cast 展宽到 tile。
 */
template <typename T, typename VEC> __aicore__ inline void SpgemmVecLoadBWindow(
    const VEC &vec, const AscendC::GlobalTensor<T> &bValGm, uint32_t minCur, uint32_t total)
{
    constexpr uint32_t kElem = SpgemmValueOps<T>::kDeferCast ? sizeof(T) : sizeof(float);
    AscendC::DataCopyExtParams cpv{
        1, static_cast<uint32_t>(total * kElem), 0, 0, 0};
    if constexpr (SpgemmValueOps<T>::kDeferCast) {
        AscendC::DataCopyPadExtParams<T> padv{false, 0, 0, 0}; AscendC::DataCopyPad(vec.narrow, bValGm[static_cast<uint64_t>(minCur)], cpv, padv);
    } else {
        AscendC::DataCopyPadExtParams<float> padv{false, 0, 0, 0}; AscendC::DataCopyPad(vec.tile, bValGm[static_cast<uint64_t>(minCur)], cpv, padv);
    }
}

/**
 * 不交叠快路径：判定各段区间两两不交叠，成立且不超 cap 时按段序整段发射。
 * 命中时通过 outCount 回传输出数并返回 true；未命中返回 false，由调用方进入归并。
 */


// 归并实现拆分为两个内部片段以降低头文件行数。
// 预处理后的 token 流与拆分前逐字相同。
#include "spgemm_merge_emit.h"
#include "spgemm_merge_row.h"


/**
 * 融合归并的段数分派。da=1/5 也固定实例化，消除 Band 回退行的
 * 动态段数索引；数值加载仍由后续路径按需完成。
 */
template <typename T, typename VEC> __aicore__ inline int32_t SpgemmMergeFusedRow(
    const AscendC::GlobalTensor<int32_t> &bColIdxGm, const AscendC::GlobalTensor<T> &bValGm,
    const SpgemmValueOps<T> &ops, SpgemmSegments &seg, const AscendC::LocalTensor<int32_t> &outCol,
    const AscendC::LocalTensor<T> &outVal, const AscendC::LocalTensor<float> &wideVal, uint32_t cap,
    const VEC &vec, uint32_t vecSlotOff, const AscendC::GlobalTensor<T> &aValGm, int32_t aBase)
{
    const uint32_t da = seg.count;
    if (da == 0) {
        return 0;
    }
    // aBase 必须传给每个实例（带路径建段不再读 A 值，冷路径按 aBase 惰性载入）。
    switch (da) {
        case 1:  return SpgemmMergeFusedRowImpl<1, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        case 2:  return SpgemmMergeFusedRowImpl<2, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        case 3:  return SpgemmMergeFusedRowImpl<3, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        case 4:  return SpgemmMergeFusedRowImpl<4, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        case 5:  return SpgemmMergeFusedRowImpl<5, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        case 6:  return SpgemmMergeFusedRowImpl<6, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
        // da=7 也拿真上下文（向量发射），DA_FIXED >= 8 的门限保证不会对 da=7 生成多余代码。
        case 7:  return SpgemmMergeFusedRowImpl<7, T, VEC>(bColIdxGm, bValGm, ops, seg, da, outCol, outVal, wideVal, cap, vec, vecSlotOff,
                                                      aValGm, aBase);
        case 8:  return SpgemmMergeFusedRowImpl<8, T, VEC>(bColIdxGm, bValGm, ops, seg, da, outCol, outVal, wideVal, cap, vec, vecSlotOff,
                                                      aValGm, aBase);
        default: return SpgemmMergeFusedRowImpl<0, T, SpgemmVecEmitNone>(bColIdxGm, bValGm, ops, seg, da,
                     outCol, outVal, wideVal, cap, SpgemmVecEmitNone{}, 0u, aValGm, aBase);
    }
}

/**
 * 加载行 i 的段游标并载入 A 值（供融合路径使用）。
 * 顺带算出 seg.apStride 和 seg.first[]。
 * @return true 表示 da 在 T1 归并能力内；false 需走 T3 路径
 */
/** B 的带状前缀描述。k < prefix 时三个描述符可由 k 解析算出。 */
struct SpgemmBandInfo {
    int32_t len = 0; int32_t stride = 0; int32_t head = 0; uint32_t prefix = 0;
};

/**
 * 通用建段循环（可处理任意输入），从快通道未命中时调用。
 */
template <typename T> __aicore__ inline bool SpgemmLoadSegmentsGmWithValGeneral(
    const AscendC::GlobalTensor<int32_t> &aColIdxGm, uint32_t rs, uint32_t len, const AscendC::GlobalTensor<int32_t> &bRowPtrGm, uint32_t K,
    const AscendC::GlobalTensor<int32_t> &bApStrideGm, const AscendC::GlobalTensor<int32_t> &bHeadGm,
    const AscendC::GlobalTensor<T> &aValGm, const SpgemmValueOps<T> &ops, const SpgemmBandInfo &band, SpgemmSegments &seg)
{
    uint32_t n = 0; uint32_t apOr = 0; uint32_t apAnd = 0xFFFFFFFFu;
    // 段长一致性累加器。
    int32_t lenFirst = 0; uint32_t lenBad = 0;
    // cur[] 的 min/max（仅复数实例需要，非复数被 DCE）。
    int32_t gMin = 0x7FFFFFFF; int32_t gMax = 0;
    // cur[] 严格升序判定（仅复数实例）：一组差值全为负 ⟺ 按位与的第 31 位为 1。
    uint32_t gAscAnd = 0xFFFFFFFFu; int32_t gPrev = -1;
    for (uint32_t i = 0; i < len; i++) {
        const int32_t k = aColIdxGm.GetValue(static_cast<uint64_t>(rs) + i);
        // 越界索引防御：host 已校验维度，此处不参与计算
        if (k < 0 || static_cast<uint32_t>(k) >= K) { continue; }
        const uint64_t kx = static_cast<uint64_t>(k);
        // 等长时解析算出段起止（见 SpgemmBandInfo）。
        int32_t bs; int32_t be;
        if (band.len > 0) {
            bs = static_cast<int32_t>(kx) * band.len; be = bs + band.len;
        } else {
            bs = bRowPtrGm.GetValue(kx); be = bRowPtrGm.GetValue(kx + 1);
        }
        // B 的该行为空，段长 0，归并自动忽略
        if (be <= bs) { continue; }
        const uint32_t ap = static_cast<uint32_t>(bApStrideGm.GetValue(kx)); apOr |= ap; apAnd &= ap;
        // 段长一致性累加。
        const int32_t rowLen = be - bs;
        if (n == 0) { lenFirst = rowLen; } else { lenBad |= static_cast<uint32_t>(rowLen ^ lenFirst); }
        seg.cur[n] = bs; seg.end[n] = be;
        // 段首从 bHead[k] 取，只缓存前 FIRST_WAYS 个。
        if (n < SPGEMM_ARCH22_FIRST_WAYS) { seg.first[n] = bHeadGm.GetValue(kx); }
        if constexpr (SpgemmValueOps<T>::kIsComplex) {
            if (bs < gMin) { gMin = bs; }
            if (bs > gMax) { gMax = bs; }
            gAscAnd &= static_cast<uint32_t>(gPrev - bs);
            gPrev = bs;
        }
        ops.LoadRaw(aValGm, rs + i, seg.valRe[n], seg.valIm[n]); n++;
    }
    seg.count = n;
    // n == 0 时留下 (INT32_MAX, 0)，但 count == 0 的行不进发射块，永不被读。
    if constexpr (SpgemmValueOps<T>::kIsComplex) {
        seg.curMin = gMin;
    // cur[] 非严格升序时令 curMax = curMin，使 c64 发射块回落标量发射。
        seg.curMax = ((gAscAnd >> 31) != 0u) ? gMax : gMin;
    }
    // 全段公差相同 ⟺ apOr == apAnd；否则置 0。
    seg.apStride = (apOr == apAnd) ? static_cast<int32_t>(apOr) : 0;
    // n == 0 时 lenFirst 仍为 0 ⇒ segLenAll = 0，与「不等长」同样使判定失败。
    seg.segLenAll = (lenBad == 0) ? lenFirst : 0;
    return true;
}

/**
 * 通用建段循环的禁止内联外壳（冷路径）。
 */
template <typename T> [[gnu::noinline]] __aicore__ bool SpgemmLoadSegmentsGmWithValColdCall(
    const AscendC::GlobalTensor<int32_t> &aColIdxGm, uint32_t rs, uint32_t len, const AscendC::GlobalTensor<int32_t> &bRowPtrGm, uint32_t K,
    const AscendC::GlobalTensor<int32_t> &bApStrideGm, const AscendC::GlobalTensor<int32_t> &bHeadGm,
    const AscendC::GlobalTensor<T> &aValGm, const SpgemmValueOps<T> &ops, const SpgemmBandInfo &band, SpgemmSegments &seg)
{
    return SpgemmLoadSegmentsGmWithValGeneral<T>(aColIdxGm, rs, len, bRowPtrGm, K, bApStrideGm, bHeadGm, aValGm, ops, band, seg);
}

/**
 * B 每行等长时的建段快通道：利用 bRowPtr[k] == k·L 解析算出段起止，免去两次 GM 标量读。
 *
 * @param aColIdxGm   A 的 colIndices（GM）
 * @param rs          A 行起始偏移
 * @param len         A 行 nnz（da）
 * @param bRowPtrGm   B 的 rowOffsets
 * @param K           B 的行数（用于越界防御）
 * @param bApStrideGm B 各行的等差公差
 * @param bHeadGm     B 各行的首列索引
 * @param aValGm      A 的 values（GM）
 * @param ops         dtype 抽象层
 * @param band        B 的带状前缀描述；prefix == 0 时逐字回落通用路径
 * @param seg         输出段游标
 */
template <typename T> __aicore__ inline bool SpgemmLoadSegmentsGmWithVal(const AscendC::GlobalTensor<int32_t> &aColIdxGm, uint32_t rs, uint32_t len,
    const AscendC::GlobalTensor<int32_t> &bRowPtrGm, uint32_t K, const AscendC::GlobalTensor<int32_t> &bApStrideGm,
    const AscendC::GlobalTensor<int32_t> &bHeadGm, const AscendC::GlobalTensor<T> &aValGm, const SpgemmValueOps<T> &ops,
    const SpgemmBandInfo &band, SpgemmSegments &seg)
{
    if (len > SPGEMM_ARCH22_MAX_WAYS) {
        return false;
    }
    // 带状快通道：band.prefix > 0 时段起止、公差、首列均可由 k 解析算出，
    // 无分支按 i 建段，末尾一次性校验；不满足则回落通用循环。
    if (band.prefix > 0 && len > 0 && len <= SPGEMM_ARCH22_FIRST_WAYS) {
        const int32_t bandLen = band.len;
        int32_t fMin = 0x7FFFFFFF;
        int32_t fMax = 0;
        // 单趟建段 + 后置越界判定。orAcc < prefix 快判全部界内，否则走精确补判。
        uint32_t orAcc = 0;
        for (uint32_t i = 0; i < len; i++) {
            const int32_t kx = aColIdxGm.GetValue(static_cast<uint64_t>(rs) + i); orAcc |= static_cast<uint32_t>(kx);
            const int32_t bs = kx * bandLen;
            if constexpr (SpgemmValueOps<T>::kIsComplex) {
                if (bs < fMin) { fMin = bs; }
                if (bs > fMax) { fMax = bs; }
            }
            seg.cur[i] = bs;
            // 带路径下不写 seg.end[]：冷分支入口处按 cur + segLenAll 就地补齐。
            seg.first[i] = band.head + kx;
            // 成功的向量发射块直接从 GM 搬 A；标量回退路径在入口补齐 A values。
        }
        uint32_t bad = 0;
        if (orAcc >= band.prefix) {
            // 冷补判：prefix <= K 是不变量，故这一句同时完成越界检查。
            for (uint32_t i = 0; i < len; i++) {
                bad |= (static_cast<uint32_t>(seg.first[i] - band.head) >= band.prefix) ? 1u : 0u;
            }
        }
        if (bad == 0) {
            seg.count = len; seg.apStride = band.stride; seg.segLenAll = bandLen;
            if constexpr (SpgemmValueOps<T>::kIsComplex) {
                seg.curMin = fMin; seg.curMax = fMax;
            }
            return true;
        }
    }
    // 快通道未命中：交给通用循环重建。
    // 非复数走 noinline 外壳（减小热函数体积），c64 保持全内联（对寄存器压力敏感）。
    if constexpr (SpgemmValueOps<T>::kIsComplex) {
        return SpgemmLoadSegmentsGmWithValGeneral<T>(aColIdxGm, rs, len, bRowPtrGm, K, bApStrideGm, bHeadGm, aValGm, ops, band, seg);
    } else {
        return SpgemmLoadSegmentsGmWithValColdCall<T>(aColIdxGm, rs, len, bRowPtrGm, K, bApStrideGm, bHeadGm, aValGm, ops, band, seg);
    }
}

/** 加载行 i 的段游标（不含 A 值，供符号阶段使用）。 */
/**
 * 加载行 i 的段游标，A 行列索引直接从 GM 标量读取。
 * @return true 表示 da 在 T1 归并能力内；false 需走 T3 路径
 */
__aicore__ inline bool SpgemmLoadSegmentsGm(const AscendC::GlobalTensor<int32_t> &aColIdxGm, uint32_t rs, uint32_t len,
    const AscendC::GlobalTensor<int32_t> &bRowPtrGm, uint32_t K, const AscendC::GlobalTensor<int32_t> &bApStrideGm,
    const AscendC::GlobalTensor<int32_t> &bHeadGm, SpgemmSegments &seg)
{
    if (len > SPGEMM_ARCH22_MAX_WAYS) {
        return false;
    }
    uint32_t n = 0; uint32_t apOr = 0; uint32_t apAnd = 0xFFFFFFFFu;
    // 段长一致性累加器。
    int32_t lenFirst = 0; uint32_t lenBad = 0;
    for (uint32_t i = 0; i < len; i++) {
        const int32_t k = aColIdxGm.GetValue(static_cast<uint64_t>(rs) + i);
        if (k < 0 || static_cast<uint32_t>(k) >= K) {
            continue;
        }
        const int32_t bs = bRowPtrGm.GetValue(static_cast<uint64_t>(k)); const int32_t be = bRowPtrGm.GetValue(static_cast<uint64_t>(k) + 1);
        if (be <= bs) {
            continue;
        }
        const uint32_t ap = static_cast<uint32_t>(bApStrideGm.GetValue(static_cast<uint64_t>(k))); apOr |= ap; apAnd &= ap;
        const int32_t rowLen = be - bs;
        if (n == 0) { lenFirst = rowLen; } else { lenBad |= static_cast<uint32_t>(rowLen ^ lenFirst); }
        seg.cur[n] = bs; seg.end[n] = be;
        if (n < SPGEMM_ARCH22_FIRST_WAYS) {
            seg.first[n] = bHeadGm.GetValue(static_cast<uint64_t>(k));
        }
        seg.valRe[n] = 0.0f; seg.valIm[n] = 0.0f; n++;
    }
    seg.count = n; seg.apStride = (apOr == apAnd) ? static_cast<int32_t>(apOr) : 0;
    // n == 0 时 lenFirst 仍为 0 ⇒ segLenAll = 0，与「不等长」同样使判定失败。
    seg.segLenAll = (lenBad == 0) ? lenFirst : 0;
    return true;
}

#endif  // SPGEMM_ARCH22_MERGE_H
