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
// UB 预算常量集中在 spmv_ub_budget.h（GEN-4.2 单一事实源，host 侧校验同源）
#include "../spmv_ub_budget.h"

// Dekker TwoProd：p = fl(a*b)，e 为 a*b - p 的精确误差项（纯 f32 运算，不依赖 FMA；
// 拆分常数 2^12+1；FMA 收缩不影响正确性）。
// 溢出防护（两道守卫）：
// 1. 拆分需先计算 4097*a / 4097*b，|a| 或 |b| > FLT_MAX/4097（≈8.3e34）时该
//    乘积溢出为 Inf，拆分将产出 NaN。此类极端输入回退为不补偿（e=0），该项退化为普通
//    乘积精度（与 fl(a*b) 一致），保证「大数×小数且积有限」的合法输入输出有限值。
// 2. 积量级守卫：即使 |a|,|b| 都在拆分安全域内，拆分后的高位积 ah*bh 仍可能溢出——
//    |ah*bh| <= |a*b|*(1+2^-11)^2，|a*b| 接近 FLT_MAX 时 ah*bh 可到 ~2^128 变 Inf
//    （反例：a=b=2^64，积 FLT_MAX 有限但 ah*bh=2^128 溢出）。|p| > FLT_MAX/2 时
//    先把 a 精确减半（乘 2^-1 不引入舍入）再拆分，最后把 p、e 精确翻倍还原，
//    保证参与拆分的积量级 <= FLT_MAX/2，ah*bh 必有限。
__aicore__ inline void SpmvTwoProd(float a, float b, float &p, float &e) {
    p = a * b;
    // 拆分安全上限：4097*|x| 不溢出当且仅当 |x| <= FLT_MAX/4097 ≈ 8.306e34，留裕量
    constexpr float SPLIT_SAFE_MAX = 8.0e34f;
    if (a >= -SPLIT_SAFE_MAX && a <= SPLIT_SAFE_MAX && b >= -SPLIT_SAFE_MAX && b <= SPLIT_SAFE_MAX) {
        // 积量级守卫：|ah*bh| <= |a*b|*(1+2^-11)^2，|a*b| > PROD_SAFE_MAX 时 ah*bh 可能溢出。
        // 将 a 精确减半后拆分（2 的幂缩放不引入舍入），p/e 翻倍还原，保证 ah*bh <= 0.5*FLT_MAX*(1+eps) 有限。
        constexpr float PROD_SAFE_MAX = 1.7e38f; // ≈ FLT_MAX/2 = 2^127*(1-2^-24)
        float sa = a;
        bool scaled = false;
        if (p > PROD_SAFE_MAX || p < -PROD_SAFE_MAX) {
            sa = a * 0.5f;
            p = sa * b;
            scaled = true;
        }
        constexpr float SPLITTER = 4097.0f; // 2^12 + 1
        float ca = SPLITTER * sa;
        float ah = ca - (ca - sa);
        float al = sa - ah;
        float cb = SPLITTER * b;
        float bh = cb - (cb - b);
        float bl = b - bh;
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        if (scaled) {
            p = p + p;  // 翻倍精确（2 的幂），p <= FLT_MAX 不溢出
            e = e + e;
        }
    }
    else {
        e = 0.0f;
    }
}

// Knuth TwoSum：s = fl(a+b)，e 为 a+b - s 的精确误差项（不要求 |a|>=|b|）
__aicore__ inline void SpmvTwoSum(float a, float b, float &s, float &e) {
    s = a + b;
    float bp = s - a;
    e = (a - (s - bp)) + (b - bp);
}

// TwoProd 守卫外提的行/窗口级安全阈值（与 SpmvTwoProd 内两道守卫同源）：
// 拆分安全要求 |a|,|b| <= FLT_MAX/4097 ≈ 8.3e34；积量级安全要求 |a*b| 使
// ah*bh <= |a*b|*(1+2^-11)^2 不溢出。谓词统一取 1.6e38（< FLT_MAX/2），为
// 上界乘法（maxVal*maxX）的舍入与 (1+2^-11)^2 留双倍裕量
constexpr float SPMV_TP_SPLIT_SAFE_MAX = 8.0e34f;
constexpr float SPMV_TP_PROD_SAFE_MAX = 1.6e38f;

// 标量绝对值（设备侧无 fabsf）
__aicore__ inline float SpmvAbsF(float v) { return v >= 0.0f ? v : -v; }

// 无守卫 Dekker TwoProd（快路径）：调用方已按行/窗口级谓词保证
// |a|,|b| <= SPMV_TP_SPLIT_SAFE_MAX 且 |a*b| <= SPMV_TP_PROD_SAFE_MAX，
// 此时拆分与高位积必有限，逐元素守卫恒不动作，逐位等价于守卫版
__aicore__ inline void SpmvTwoProdFast(float a, float b, float &p, float &e) {
    p = a * b;
    constexpr float SPLITTER = 4097.0f; // 2^12 + 1
    float ca = SPLITTER * a;
    float ah = ca - (ca - a);
    float al = a - ah;
    float cb = SPLITTER * b;
    float bh = cb - (cb - b);
    float bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// max|src[0..count)| 矢量归约（max 与 -min 取大）。dstTmp 为至少 256 floats 的
// 死区（dst 用前 8，tmp 用后 248，与 ReduceProdSegment 的 bufReduceWk 用法同口径）；
// count <= 4096（归约 tmp 的实测安全上界，与长行段 ReduceSum 一致）
__aicore__ inline float SpmvVecMaxAbs(const AscendC::LocalTensor<float> &src, int32_t count,
                                      const AscendC::LocalTensor<float> &dstTmp) {
    AscendC::ReduceMax<float>(dstTmp, src, dstTmp[8], count, false);
    // 矢量写 dst → 标量读
    AscendC::PipeBarrier<PIPE_V>();
    float mx = dstTmp.GetValue(0);
    // 标量读 → 矢量写 dst（WAR）
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMin<float>(dstTmp, src, dstTmp[8], count, false);
    AscendC::PipeBarrier<PIPE_V>();
    float mn = dstTmp.GetValue(0);
// 标量读 → 后续矢量覆写 dstTmp（WAR）
    AscendC::PipeBarrier<PIPE_V>();
    return mx > -mn ? mx : -mn;
}

// TwoProd 分派：GUARDED=true 走逐点守卫版，false 走无守卫快路径（调用方谓词
// 已保证安全）。让 fast/slow 共用同一份主体代码，避免双份实现的重复块
template <bool GUARDED>
__aicore__ inline void SpmvTwoProdSel(float a, float b, float &p, float &e) {
    if constexpr (GUARDED) {
        SpmvTwoProd(a, b, p, e);
    }
    else {
        SpmvTwoProdFast(a, b, p, e);
    }
}

// DS 双字窗口单槽位 TwoSum 合并：(acc[slot], accLo[slot]) += (h, l)
__aicore__ inline void SpmvDsMerge(__ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
                                   uint32_t slot, float h, float l) {
    float s, e;
    SpmvTwoSum(accPtr[slot], h, s, e);
    accLoPtr[slot] += e + l;
    accPtr[slot] = s;
}

// N 路径 tile 级守卫谓词：返回 tileSafe（本 tile 全部非零元可否走无守卫快路径），
// rowSafeMax 输出行级阈值（max|x| 超拆分安全域时为 -1，禁止一切快路径（含零值元））；
// redWk 为归约死区工作区（须在 Gather/ReduceSum 之前使用）
__aicore__ inline bool SpmvTileSafeN(const AscendC::LocalTensor<float> &valsF, int32_t valid,
                                     float maxAbsX, const AscendC::LocalTensor<float> &redWk,
                                     float &rowSafeMax) {
    float tileMaxV = SpmvVecMaxAbs(valsF, valid, redWk);
    // max|x| 超拆分安全域时置 -1 哨兵：tileMaxV/rowMax 恒 >= 0，<= -1 恒假，
    // tile 级与行级快路径全禁——含零值元：快路径拆分需 4097*b 有限，b 越限时
    // 即使 a=0 也会 Inf-Inf 产出 NaN，必须走守卫路径（SPLIT 守卫回退 e=0、
    // p=fl(a*b)=0，结果正确）
    rowSafeMax = -1.0f;
    if (maxAbsX <= SPMV_TP_SPLIT_SAFE_MAX) {
        rowSafeMax = SPMV_TP_SPLIT_SAFE_MAX;
        if (maxAbsX > 0.0f) {
            float lim = SPMV_TP_PROD_SAFE_MAX / maxAbsX;
            rowSafeMax = lim < rowSafeMax ? lim : rowSafeMax;
        }
    }
    return tileMaxV <= rowSafeMax;
}

// T 路径（f32）段级守卫谓词：返回 tileSafe（本段可否走无守卫快路径）；
// grpMaxX 为本组 x 窗口 max|x|，absAlpha = |alpha|。判据与逐点守卫一致：
// pv = max|val|*max|x| 覆盖 val*x 的拆分与积量级，pv*|alpha| 覆盖第二级
// alpha*p1（上界乘法溢出为 Inf 时判不安全，方向正确）
__aicore__ inline bool SpmvTileSafeT(const AscendC::LocalTensor<float> &valsF, int32_t valid,
                                     float grpMaxX, float absAlpha,
                                     const AscendC::LocalTensor<float> &redWk) {
    float tileMaxV = SpmvVecMaxAbs(valsF, valid, redWk);
    float pv = tileMaxV * grpMaxX;
    return tileMaxV <= SPMV_TP_SPLIT_SAFE_MAX && grpMaxX <= SPMV_TP_SPLIT_SAFE_MAX &&
           pv <= SPMV_TP_SPLIT_SAFE_MAX && absAlpha <= SPMV_TP_SPLIT_SAFE_MAX &&
           pv * absAlpha <= SPMV_TP_PROD_SAFE_MAX;
}

// 单短行段 Dot2 补偿累加主体（Dekker 精确积 + Knuth TwoSum 进 (s, c)）；
// GUARDED=false 为无守卫快路径（调用方谓词已保证本行全部乘积安全），
// 快/慢路径对正常输入逐位一致（守卫恒不动作）
template <bool MULTI_WIN, bool GUARDED>
__aicore__ inline void SpmvDot2RowSeg(int32_t lo, int32_t hi, int32_t t0,
                                      const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr,
                                      const __ubuf__ float *xPtr, uint32_t winStart, uint32_t winEnd,
                                      float &s, float &c) {
    for (int32_t j = lo; j < hi; j++) {
        int32_t col = colPtr[j - t0];
        if constexpr (MULTI_WIN) {
            if (static_cast<uint32_t>(col) < winStart ||
                static_cast<uint32_t>(col) >= winEnd)
                continue;
        }
        float xv;
        if constexpr (MULTI_WIN) {
            xv = xPtr[col - static_cast<int32_t>(winStart)];
        }
        else {
            xv = xPtr[col];
        }
        float p, pErr;
        if constexpr (GUARDED) {
            SpmvTwoProd(valPtr[j - t0], xv, p, pErr);
        }
        else {
            SpmvTwoProdFast(valPtr[j - t0], xv, p, pErr);
        }
        float t = s + p;
        float bp = t - s;
        c += (s - (t - bp)) + (p - bp);
        c += pErr;
        s = t;
    }
}

// -------------------------------------------------------------------
// 各 kernel 共用的 Init / 加载 / 行段走查原语（结构抽取，运算与次序不变）
// -------------------------------------------------------------------

// 四个 kernel 类 Init 的公共标量赋值
__aicore__ inline void SpmvInitCommon(uint32_t totalRowsNum, uint32_t totalColsNum,
                                      uint32_t indexBase, uint32_t readY,
                                      uint32_t &outRows, uint32_t &outCols,
                                      uint32_t &outIndexBase, uint32_t &outReadY) {
    outRows = totalRowsNum;
    outCols = totalColsNum;
    outIndexBase = indexBase;
    outReadY = readY;
}

// c64 两个 kernel 类 Init 的 alpha/beta 复数标量赋值
__aicore__ inline void SpmvInitC64Scalars(float alphaRe, float alphaIm, float betaRe, float betaIm,
                                          float &outAr, float &outAi, float &outBr, float &outBi) {
    outAr = alphaRe;
    outAi = alphaIm;
    outBr = betaRe;
    outBi = betaIm;
}

// 均分 total 到各核：前 temp 个核各多拿 1 份（行切分与列切分共用）
__aicore__ inline void SpmvSplitBlockRange(uint32_t total, uint32_t blockNum,
                                           uint32_t &count, uint32_t &start) {
    if (blockNum == 0) {
        count = 0;
        start = 0;
        return;
    }
    uint32_t temp = total % blockNum;
    if (AscendC::GetBlockIdx() < temp) {
        count = total / blockNum + 1;
        start = count * AscendC::GetBlockIdx();
    }
    else {
        count = total / blockNum;
        start = count * AscendC::GetBlockIdx() + temp;
    }
}

// rowPtr 绑定 + 0 基 nnz 区间 [startValIdx, endValIdx)（rowPtr 含 indexBase，
// 统一减去得到 0 基 nnz 偏移）
__aicore__ inline void SpmvBindRowRange(AscendC::GlobalTensor<int32_t> &rowPtrGm,
                                        GM_ADDR csrRowPtr, uint32_t startRow, uint32_t blockRowNum,
                                        uint32_t indexBase,
                                        uint32_t &startValIdx, uint32_t &endValIdx) {
    if (blockRowNum > 0) {
        rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + startRow, blockRowNum + 1);
        int32_t s = rowPtrGm(0) - static_cast<int32_t>(indexBase);
        int32_t e = rowPtrGm(blockRowNum) - static_cast<int32_t>(indexBase);
        startValIdx = (e >= s) ? static_cast<uint32_t>(s) : 0;
        endValIdx = (e >= s) ? static_cast<uint32_t>(e) : 0;
    } else {
        startValIdx = 0;
        endValIdx = 0;
    }
}

// ValT -> CompT 转换：16bit 走矢量 Cast（禁止标量转换）；USE_HALF_I8 时 int8->float
// 走 int8->half->float 两步矢量 Cast（int8 值在半精度下精确无损失，tmpHalf 复用积
// 缓冲，此时积缓冲尚未启用）；其余整型走标量（数值小、无精度损失）
template <typename CompT, typename ValT, bool USE_HALF_I8>
__aicore__ inline void SpmvConvertVals(const AscendC::LocalTensor<ValT> &raw,
                                       const AscendC::LocalTensor<CompT> &dst, uint32_t count,
                                       const AscendC::LocalTensor<half> &tmpHalf) {
    if constexpr (std::is_same_v<ValT, CompT>) {
        return; // 已直接 DMA 到 comp 缓冲
    }
    else if constexpr (std::is_same_v<ValT, half> || std::is_same_v<ValT, bfloat16_t>) {
        AscendC::Cast<CompT, ValT>(dst, raw, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    else if constexpr (USE_HALF_I8 && std::is_same_v<ValT, int8_t> &&
                       std::is_same_v<CompT, float>) {
        for (uint32_t off = 0; off < count; off += SPMV_TILE_ELEMS) {
            uint32_t n = count - off < SPMV_TILE_ELEMS ? count - off : SPMV_TILE_ELEMS;
            AscendC::Cast<half, int8_t>(tmpHalf, raw[off], AscendC::RoundMode::CAST_NONE, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<CompT, half>(dst[off], tmpHalf, AscendC::RoundMode::CAST_NONE, n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    else {
        // 裸 __ubuf__ 指针：规避 GetValue/SetValue 的 pipe 串行化
        __ubuf__ ValT *src = reinterpret_cast<__ubuf__ ValT *>(raw.GetPhyAddr());
        __ubuf__ CompT *dstPtr = reinterpret_cast<__ubuf__ CompT *>(dst.GetPhyAddr());
        for (int32_t i = 0; i < static_cast<int32_t>(count); i++) {
            dstPtr[i] = static_cast<CompT>(src[i]);
        }
    }
}

// 行组 rowPtr 批量 DMA 入 UB（groupCount+1 个）
__aicore__ inline void SpmvLoadGroupRowPtr(AscendC::GlobalTensor<int32_t> &rowPtrGm,
                                           const AscendC::LocalTensor<int32_t> &rowPtr,
                                           uint32_t groupStart, uint32_t groupCount) {
    // 实验：完全去屏障（前导 WAR 无屏障 +  trailing PIPE_MTE2）
    AscendC::DataCopyExtParams cpRp{1, (uint32_t)((groupCount + 1) * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padRp{false, 0, 0, 0};
    AscendC::DataCopyPad(rowPtr, rowPtrGm[groupStart], cpRp, padRp);
    AscendC::PipeBarrier<PIPE_MTE2>();
}

// 本组 nnz 全局区间 [groupNnz0, groupNnz1)（rowPtr 含 indexBase，统一减 1）
__aicore__ inline void SpmvGroupNnzRange(const AscendC::LocalTensor<int32_t> &rowPtr,
                                         uint32_t groupCount, uint32_t indexBase,
                                         int32_t &groupNnz0, int32_t &groupNnz1) {
    groupNnz0 = rowPtr.GetValue(0) - static_cast<int32_t>(indexBase);
    groupNnz1 = rowPtr.GetValue(groupCount) - static_cast<int32_t>(indexBase);
    if (groupNnz1 < groupNnz0)
        groupNnz1 = groupNnz0;
}

// tile 步进：valid 截断到 SPMV_TILE_ELEMS，返回 t1 = t0 + valid
__aicore__ inline int32_t SpmvBeginTile(int32_t t0, int32_t groupNnz1, int32_t &valid) {
    valid = groupNnz1 - t0;
    if (valid > static_cast<int32_t>(SPMV_TILE_ELEMS))
        valid = SPMV_TILE_ELEMS;
    return t0 + valid;
}

// 一拍 colIdx tile 批量 DMA 入 UB（前导 barrier 等上一拍矢量阶段完成）
__aicore__ inline void SpmvLoadColIdxTile(AscendC::GlobalTensor<int32_t> &colGm,
                                          const AscendC::LocalTensor<int32_t> &colIdx,
                                          int32_t t0, uint32_t startValIdx, int32_t valid) {
    // WAR：等上一拍矢量消费（Adds/gather）完成后再由 MTE2 覆写；标量游标读随程序序已完成
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopyExtParams cpI{1, (uint32_t)(valid * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
    AscendC::DataCopyPad(colIdx, colGm[t0 - static_cast<int32_t>(startValIdx)], cpI, padI);
}

// 一拍 values tile 批量 DMA 入 UB（同类型直达 comp 缓冲，异类型入 raw 缓冲待转换）
template <typename ValT, typename CompT>
__aicore__ inline void SpmvLoadValsTile(AscendC::GlobalTensor<ValT> &valGm,
                                        const AscendC::LocalTensor<CompT> &valsC,
                                        const AscendC::LocalTensor<ValT> &valsRaw,
                                        int32_t t0, uint32_t startValIdx, int32_t valid) {
    if constexpr (std::is_same_v<ValT, CompT>) {
        AscendC::DataCopyExtParams cpV{1, (uint32_t)(valid * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
        AscendC::DataCopyPad(valsC, valGm[t0 - static_cast<int32_t>(startValIdx)], cpV, padV);
    }
    else {
        AscendC::DataCopyExtParams cpV{1, (uint32_t)(valid * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
        AscendC::DataCopyPad(valsRaw, valGm[t0 - static_cast<int32_t>(startValIdx)], cpV, padV);
    }
}

// indexBase == 1 时列索引统一减 1
__aicore__ inline void SpmvApplyIndexBase(const AscendC::LocalTensor<int32_t> &colIdx,
                                          uint32_t indexBase, int32_t valid) {
    if (indexBase == 1) {
        AscendC::Adds(colIdx, colIdx, -1, valid);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// 加载一拍 colIdx / values tile 并完成 ValT->CompT 转换（indexBase 一并处理）
template <typename CompT, typename ValT, bool USE_HALF_I8>
__aicore__ inline void SpmvLoadNzTile(AscendC::GlobalTensor<int32_t> &colGm,
                                      AscendC::GlobalTensor<ValT> &valGm,
                                      const AscendC::LocalTensor<int32_t> &colIdx,
                                      const AscendC::LocalTensor<CompT> &valsC,
                                      const AscendC::LocalTensor<ValT> &valsRaw,
                                      const AscendC::LocalTensor<half> &tmpHalf,
                                      int32_t t0, uint32_t startValIdx, uint32_t indexBase,
                                      int32_t valid) {
    SpmvLoadColIdxTile(colGm, colIdx, t0, startValIdx, valid);
    SpmvLoadValsTile<ValT, CompT>(valGm, valsC, valsRaw, t0, startValIdx, valid);
    // MTE2（colIdx/vals 入 UB）→矢量 Adds/Cast：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    SpmvApplyIndexBase(colIdx, indexBase, valid);
    SpmvConvertVals<CompT, ValT, USE_HALF_I8>(valsRaw, valsC, valid, tmpHalf);
    // 矢量写（转换/减基）完成后，后续矢量 gather 与标量游标读才可见
    AscendC::PipeBarrier<PIPE_V>();
}

// 行段游标：按 rowPtr 走查与 [t0, t1) 相交的非空行段（先跳过整段在 tile 前的行，
// 再逐行返回 lo/hi 钳位到 tile 的非空段）。各 kernel 的行段循环共用，保证走查
// 次序、边界判定与原来逐处内联的 while 循环完全一致
struct SpmvRowSegCursor {
    const __ubuf__ int32_t *rpPtr;
    uint32_t indexBase;
    uint32_t r;
    uint32_t groupCount;
    int32_t t0;
    int32_t t1;
    int32_t lo;
    int32_t hi;

    // 跳过本 tile 之前（含空行）的行
    __aicore__ inline void Init(const __ubuf__ int32_t *rp, uint32_t base, uint32_t row,
                                uint32_t count, int32_t tile0, int32_t tile1) {
        rpPtr = rp;
        indexBase = base;
        groupCount = count;
        t0 = tile0;
        t1 = tile1;
        r = row;
        while (r < groupCount &&
               rpPtr[r + 1] - static_cast<int32_t>(indexBase) <= t0) {
            r++;
        }
    }

    // 取下一个与 [t0, t1) 相交的非空行段（行号经 rowOut 返回，区间为 lo/hi）；
    // 剩余行均在本 tile 之后或为空段时返回 false
    __aicore__ inline bool Next(uint32_t &rowOut) {
        while (r < groupCount) {
            int32_t rs = rpPtr[r] - static_cast<int32_t>(indexBase);
            if (rs >= t1)
                return false;
            int32_t re = rpPtr[r + 1] - static_cast<int32_t>(indexBase);
            lo = rs > t0 ? rs : t0;
            hi = re < t1 ? re : t1;
            rowOut = r;
            r++;
            if (hi > lo)
                return true;
        }
        return false;
    }
};

// 行组头：组行数（截断到 SPMV_ROW_GROUP）与本组 nnz 全局区间
struct SpmvGroupHead {
    uint32_t groupCount;
    int32_t groupNnz0;
    int32_t groupNnz1;
};

// 行段游标遍历本 tile 全部行段，逐段交给 kernel.AccumulateSeg 累加
//（f32 T 与 c64 T 共用；indexBase 显式传入，AccumulateSeg 由各 kernel 类实现）
template <typename KT, typename CompT>
__aicore__ inline void SpmvAccumulateTileSegs(KT &kernel, uint32_t indexBase,
    const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
    const __ubuf__ CompT *valPtr, const __ubuf__ CompT *xRowPtr, __ubuf__ CompT *yAccPtr,
    uint32_t row, uint32_t groupCount, int32_t t0, int32_t t1,
    uint32_t winStart, uint32_t winEnd) {
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, indexBase, row, groupCount, t0, t1);
    uint32_t r;
    while (cur.Next(r)) {
        kernel.AccumulateSeg(r, cur.lo, cur.hi, colPtr, valPtr, xRowPtr, yAccPtr,
                             t0, winStart, winEnd);
    }
}

// 同上，逐段补偿累加进 (acc, accLo) 双字窗口（DS 路径共用）。
// TwoProd 守卫外提：tileSafe 为真（tile 级谓词已判定安全）时全部行段走无守卫
// 快路径；否则逐行段由 kernel.RowDsSafe 判定（行内 max|val| 与该行的 x 值合成
// 上界），按行选快/慢路径；absAlphaMax 为 |alpha|（复数取实虚部最大者）
template <typename KT, typename CompT>
__aicore__ inline void SpmvAccumulateTileSegsDs(KT &kernel, uint32_t indexBase,
    const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
    const __ubuf__ CompT *valPtr, const __ubuf__ CompT *xRowPtr,
    __ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
    uint32_t row, uint32_t groupCount, int32_t t0, int32_t t1,
    uint32_t winStart, uint32_t winEnd, bool tileSafe, float absAlphaMax) {
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, indexBase, row, groupCount, t0, t1);
    uint32_t r;
    while (cur.Next(r)) {
        if (tileSafe || kernel.RowDsSafe(r, cur.lo, cur.hi, colPtr, valPtr, xRowPtr,
                                         t0, winStart, winEnd, absAlphaMax)) {
            kernel.template AccumulateSegDs<false>(r, cur.lo, cur.hi, colPtr, valPtr, xRowPtr,
                                                   accPtr, accLoPtr, t0, winStart, winEnd);
        }
        else {
            kernel.template AccumulateSegDs<true>(r, cur.lo, cur.hi, colPtr, valPtr, xRowPtr,
                                                  accPtr, accLoPtr, t0, winStart, winEnd);
        }
    }
}

// c64 行段 x 值：复数输入取交错 float2（xr, xi），实数输入取单 float（xi 恒 0）
template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvFetchXRI(const __ubuf__ float *xRowPtr, uint32_t r,
                                    float &xr, float &xi) {
    xr = 0.0f;
    xi = 0.0f;
    if constexpr (INPUT_COMPLEX) {
        xr = xRowPtr[2 * r];
        xi = xRowPtr[2 * r + 1];
    }
    else {
        xr = xRowPtr[r];
    }
}

// 16bit 舍入邻居选取：r 恰落舍入中点附近时按 d = r - yf 方向取相邻位型 nb
//（yBits 读入、nbBits 输出）；±0 取最小非零邻居，inf/nan 不修正
template <typename OutT>
__aicore__ inline void SpmvRnePickNeighbors(uint32_t chunkSize, const __ubuf__ float *loPtr,
                                            const __ubuf__ float *stgPtr, const __ubuf__ float *yfPtr,
                                            __ubuf__ uint16_t *yBits, __ubuf__ uint16_t *nbBits) {
    constexpr uint16_t EXP_ONES = std::is_same_v<OutT, half> ? 0x7C00 : 0x7F80;
    for (uint32_t j = 0; j < chunkSize; j++) {
        uint16_t b = yBits[j];
        uint16_t nb = b;
        if (loPtr[j] != 0.0f) {
            float d = stgPtr[j] - yfPtr[j];
            if (d != 0.0f) {
                if ((b & 0x7FFF) == 0) {
                    nb = (d > 0.0f) ? 0x0001 : 0x8001;    // ±0 的最小非零邻居
                } else {
                    bool positive = (b & 0x8000) == 0;
                    nb = ((d > 0.0f) == positive) ? b + 1 : b - 1;
                }
                // inf/nan 不修正（f16 指数全 1 为 0x7C00，bf16 为 0x7F80）
                if ((b & 0x7FFF) >= EXP_ONES || (nb & 0x7FFF) >= EXP_ONES)
                    nb = b;
            }
        }
        nbBits[j] = nb;
    }
}

// 16bit 舍入中点重判：仅当 Cast 输入 r 恰为 16bit 舍入中点时，RNE 的平局取偶
// 不等于真值 s+lo 的方向，用双字重算 d3 = (s - mid) + lo 定方向；r 非中点时
// Cast 结果即为正确舍入（f32 噪声带 ~2^-24，可忽略）
template <typename OutT>
__aicore__ inline void SpmvRneMidpointRecheck(uint32_t chunkSize, const __ubuf__ float *hiPtr,
                                               const __ubuf__ float *loPtr, const __ubuf__ float *stgPtr,
                                               const __ubuf__ float *yfPtr, const __ubuf__ float *nfPtr,
                                               __ubuf__ uint16_t *yBits, const __ubuf__ uint16_t *nbBits) {
    for (uint32_t j = 0; j < chunkSize; j++) {
        uint16_t b = yBits[j];
        uint16_t nb = nbBits[j];
        if (nb == b)
            continue;
        float mid = (yfPtr[j] + nfPtr[j]) * 0.5f;   // 中点，f32 精确可表示
        if (stgPtr[j] != mid)
            continue;
        float d3 = (hiPtr[j] - mid) + loPtr[j];     // 真值 - 中点（双字）
        if (d3 == 0.0f)
            continue;                               // 真值恰在中点，取偶已对
        bool up = d3 > 0.0f;
        bool nfUp = nfPtr[j] > yfPtr[j];
        yBits[j] = (up == nfUp) ? nb : b;
    }
}

/**
 * @brief CSR 格式稀疏矩阵-稠密向量乘法（SpMV，非转置）AscendC 算子
 *
 * 行并行：每个 AI Core 处理若干整行。性能要点：
 * 1. x 按列窗口预加载到 UB（窗口内 gather 走 UB 标量读，避免 GM gather 延迟）；
 *    x 超过窗口预算时按窗口多遍扫描，每遍只累加落在窗口内的非零元。
 * 2. rowPtr/colInd/values 均按组/块批量 DMA 入 UB，行内按 (行, tile) 双指针走查，
 *    消除逐行的小块 DataCopyPad 与队列同步开销。
 * 3. 行结果先聚合到 UB 累加器，末尾统一 alpha*acc + beta*y 后整块写出。
 * indexBase 为 1 时列索引统一减 1；readY 为 0（beta==0）时不读 y，直接覆写。
 * CompT == int32_t 时采用整数标量累加，保证 INT8->INT32 逐元素精确。
 * CompT == float 时采用 Dot2 补偿累加（Dekker 精确积 + Knuth TwoSum，补偿量
 * 存 bufAccLo）：f32 树形/顺序归约的舍入误差相对部分和量级，在正负消减
 * （golden 近零）场景下会超出 ATK 严格判定（rtol=1e-5, atol=1e-8）；Dot2 将
 * 点积误差压至 ~2^-45 量级，仅剩最后一次输出舍入。
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernel {
public:
    // x 窗口元素数：raw + comp 两份缓冲合计不超过预算。
    // 同类型无 raw，预算 128KB；CompT==float 的异类型给足 112KB（28672 列单窗口
    // 覆盖 P 场景；ValT>=4B 时 valsRaw 占 16KB、窗口降为 88KB 防 UB 溢出），
    // raw 改用小分块循环加载；其余（i8->i32）96KB。
    static constexpr uint32_t XWIN_ELEMS =
        (std::is_same_v<ValT, CompT> ? 128 * 1024
                                     : (std::is_same_v<CompT, float>
                                            ? (sizeof(ValT) >= 4 ? 88 * 1024 : 112 * 1024)
                                            : SPMV_XWIN_BYTES)) /
            (sizeof(CompT) + (std::is_same_v<ValT, CompT> || std::is_same_v<CompT, float> ? 0 : sizeof(ValT))) /
            256 * 256;

    __aicore__ inline SpmvKernel(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta, uint32_t indexBase, uint32_t readY);
    __aicore__ inline void Process();

    // CompT==float 异类型时 x 窗口的 DMA 分块大小（元素）
    static constexpr uint32_t XRAW_CHUNK = 2048;

private:
    __aicore__ inline void BeginGroup(uint32_t groupStart, uint32_t groupCount);
    __aicore__ inline void LoadNzTile(int32_t t0, int32_t valid);
    __aicore__ inline void AccumulateTile(uint32_t &row, int32_t t0, int32_t t1, int32_t valid,
                                          uint32_t groupCount, uint32_t winStart, uint32_t winEnd,
                                          uint32_t winElems, uint32_t winCountTotal);
    // Pass 1：短行段标量 Dot2（Dekker 精确积 + Knuth TwoSum）；MULTI_WIN 时仅累加
    // 落在当前窗口内的非零元。tileSafe 为真（tile 级谓词已判定安全）时全行走
    // 无守卫快路径；否则逐短行扫描行内 max|val| 与 rowSafeMax 比较，按行选
    // 快/慢路径（慢路径 = 带逐点守卫的 SpmvTwoProd）
    template <bool MULTI_WIN>
    __aicore__ inline void Dot2ShortRows(const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
                                         const __ubuf__ float *valPtr, __ubuf__ float *xPtr,
                                         __ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
                                         int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount,
                                         uint32_t winStart, uint32_t winEnd,
                                         float rowSafeMax, bool tileSafe);
    // 矢量 Gather 聚集 x 至 prod：SINGLE_WIN 无钳位；多窗口 t = c - winStart 钳位
    // 到 [0, winElems-1] 后 Gather，出窗元素由「钳位前后相等」掩码 Select 置 0
    template <bool SINGLE_WIN>
    __aicore__ inline void GatherXWindow(AscendC::LocalTensor<CompT> prod,
                                         const AscendC::LocalTensor<int32_t> &colIdx,
                                         const AscendC::LocalTensor<CompT> &xWin,
                                         int32_t valid, uint32_t winStart, uint32_t winElems);
    // Pass 2：长行段（>= SPMV_LONG_ROW_MIN）矢量积 + 树形归约，TwoSum 进 (acc, accLo)
    __aicore__ inline void ReduceLongRows(const __ubuf__ int32_t *rpPtr, __ubuf__ float *accPtr,
                                          __ubuf__ float *accLoPtr,
                                          const AscendC::LocalTensor<CompT> &prod,
                                          int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount);
    // INT32 精确路径：标量整数乘加，与归约顺序无关（exact match）
    __aicore__ inline void AccumulateTileInt32(const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
                                               const AscendC::LocalTensor<CompT> &valsC,
                                               const AscendC::LocalTensor<CompT> &xWin,
                                               __ubuf__ CompT *accPtr, int32_t t0, int32_t t1,
                                               uint32_t row, uint32_t groupCount,
                                               uint32_t winStart, uint32_t winEnd);
    __aicore__ inline void LoadXWindow(uint32_t winStart, uint32_t winCount);
    __aicore__ inline void ConvertVals(const AscendC::LocalTensor<ValT> &raw,
                                       const AscendC::LocalTensor<CompT> &dst, uint32_t count);
    __aicore__ inline void FinalizeGroup(uint32_t groupStart, uint32_t groupCount);
    __aicore__ inline CompT ReduceProdSegment(const AscendC::LocalTensor<CompT> &prod,
                                              int32_t start, int32_t len);
    // 一次性矢量归约全向量 max|x|（仅 CompT==float 调用；供 Dot2 守卫外提谓词）
    __aicore__ inline float ScanMaxAbsX();

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRowPtr, bufAcc, bufAccLo, bufColIdx;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufValsC, bufValsRaw;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXWin, bufXRaw;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYRaw, bufYC, bufYOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufProd, bufReduceWk;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;
    uint32_t indexBase;
    uint32_t readY;
    float maxAbsX; // 全向量 max|x|（ScanMaxAbsX 填充，仅浮点路径使用）
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta, uint32_t indexBase, uint32_t readY) {
    SpmvInitCommon(totalRowsNum, totalColsNum, indexBase, readY,
                   this->totalRowsNum, this->totalColsNum, this->indexBase, this->readY);
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0) {
        this->blockRowNum = 0;
        return;
    }
    // 均分行：前 tempRowNum 个核各多拿一行
    SpmvSplitBlockRange(totalRowsNum, blockNum, this->blockRowNum, this->startRow);

    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalColsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec + this->startRow, this->blockRowNum);

    SpmvBindRowRange(csrRowPtrGm, csrRowPtr, this->startRow, this->blockRowNum, this->indexBase,
                     this->startValIdx, this->endValIdx);
    if (this->blockRowNum > 0) {
        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + this->startValIdx,
                                    this->endValIdx - this->startValIdx);
        csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + this->startValIdx,
                                 this->endValIdx - this->startValIdx);
    }

    pipe.InitBuffer(bufRowPtr, (SPMV_ROW_GROUP + 1) * sizeof(int32_t));
    pipe.InitBuffer(bufAcc, SPMV_ROW_GROUP * sizeof(CompT));
    // Dot2 补偿累加器（仅浮点路径使用，int32 走标量精确累加不需要）
    pipe.InitBuffer(bufAccLo,
        std::is_same_v<CompT, int32_t> ? 32 : SPMV_ROW_GROUP * sizeof(CompT));
    pipe.InitBuffer(bufColIdx, SPMV_TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufValsC, SPMV_TILE_ELEMS * sizeof(CompT));
    pipe.InitBuffer(bufValsRaw,
        std::is_same_v<ValT, CompT> ? 32 : SPMV_TILE_ELEMS * sizeof(ValT));
    pipe.InitBuffer(bufXWin, XWIN_ELEMS * sizeof(CompT));
    // CompT==float 的异类型 x 按 XRAW_CHUNK 分块加载转换，raw 缓冲只需一个分块
    pipe.InitBuffer(bufXRaw,
        std::is_same_v<ValT, CompT> ? 32
                                    : (std::is_same_v<CompT, float> ? XRAW_CHUNK * sizeof(ValT)
                                                                    : XWIN_ELEMS * sizeof(ValT)));
    pipe.InitBuffer(bufYRaw, SPMV_ROW_GROUP * sizeof(OutT));
    pipe.InitBuffer(bufYC,
        std::is_same_v<CompT, OutT> ? 32 : SPMV_ROW_GROUP * sizeof(CompT));
    pipe.InitBuffer(bufYOut,
        std::is_same_v<CompT, OutT> ? 32 : SPMV_ROW_GROUP * sizeof(OutT));
    // 积缓冲 + 归约工作区仅浮点路径需要（INT32 走标量精确累加）
    pipe.InitBuffer(bufProd,
        std::is_same_v<CompT, int32_t> ? 32 : SPMV_TILE_ELEMS * sizeof(CompT));
    pipe.InitBuffer(bufReduceWk,
        std::is_same_v<CompT, int32_t> ? 32 : 1024);
}

// ValT -> CompT 转换（委托共用原语；int8->float 走两步矢量 Cast，tmpHalf 复用积缓冲）
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::ConvertVals(
    const AscendC::LocalTensor<ValT> &raw, const AscendC::LocalTensor<CompT> &dst, uint32_t count) {
    SpmvConvertVals<CompT, ValT, true>(raw, dst, count, bufProd.Get<half>());
}

// 长行段求和：头部标量剥离到 32B 对齐后调用矢量 ReduceSum（树形归约，长尾行精度
// 比逐元素顺序累加更接近 fp64 golden），dst/tmp 复用 bufReduceWk（前 8 元素为 dst）
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline CompT SpmvKernel<CompT, ValT, OutT>::ReduceProdSegment(
    const AscendC::LocalTensor<CompT> &prod, int32_t start, int32_t len) {
    __ubuf__ CompT *prodPtr = reinterpret_cast<__ubuf__ CompT *>(prod.GetPhyAddr());
    constexpr int32_t ALIGN_ELEMS = 32 / sizeof(CompT);
    int32_t head = (ALIGN_ELEMS - (start & (ALIGN_ELEMS - 1))) & (ALIGN_ELEMS - 1);
    if (head > len)
        head = len;
    CompT sum = static_cast<CompT>(0);
    for (int32_t k = 0; k < head; k++) {
        sum += prodPtr[start + k];
    }
    int32_t rest = len - head;
    if (rest > 0) {
        AscendC::LocalTensor<CompT> wk = bufReduceWk.Get<CompT>();
        AscendC::LocalTensor<CompT> wkTmp = wk[ALIGN_ELEMS];
        // prod 的前序矢量写与 ReduceSum 同管程序序；标量头段读随程序序完成，无需屏障
        AscendC::ReduceSum(wk, prod[start + head], wkTmp, rest);
        // 矢量写 wk → 标量读 wkPtr[0]
        AscendC::PipeBarrier<PIPE_V>();
        __ubuf__ CompT *wkPtr = reinterpret_cast<__ubuf__ CompT *>(wk.GetPhyAddr());
        sum += wkPtr[0];
    }
    return sum;
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::LoadXWindow(uint32_t winStart, uint32_t winCount) {
    // WAR：等上一窗口矢量 Gather/标量 Dot2 读完 xWin/xRaw，再由 MTE2 覆写
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<ValT, CompT>) {
        AscendC::LocalTensor<CompT> xWin = bufXWin.Get<CompT>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(winCount * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xWin, xVecGm[winStart], cp, pad);
        // MTE2→矢量 Gather：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    else if constexpr (std::is_same_v<CompT, float>) {
        // xWin 足够大（预算 112KB），raw 用小分块循环加载并转换，避免双份大缓冲
        AscendC::LocalTensor<ValT> xRaw = bufXRaw.Get<ValT>();
        AscendC::LocalTensor<CompT> xWin = bufXWin.Get<CompT>();
        for (uint32_t off = 0; off < winCount; off += XRAW_CHUNK) {
            uint32_t n = winCount - off < XRAW_CHUNK ? winCount - off : XRAW_CHUNK;
            AscendC::DataCopyExtParams cp{1, (uint32_t)(n * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(xRaw, xVecGm[winStart + off], cp, pad);
            // MTE2→矢量 ConvertVals：同上，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            ConvertVals(xRaw, xWin[off], n);
            // 矢量读 xRaw 毕（下一拍 MTE2 覆写 xRaw 的 WAR）+ 矢量写 xWin 供后续 Gather
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    else {
        AscendC::LocalTensor<ValT> xRaw = bufXRaw.Get<ValT>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(winCount * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xRaw, xVecGm[winStart], cp, pad);
        // MTE2→矢量 ConvertVals：同上，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<CompT> xWin = bufXWin.Get<CompT>();
        ConvertVals(xRaw, xWin, winCount);
        // 矢量写 xWin → 后续矢量 Gather/标量 Dot2 读
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// 一次性矢量归约全向量 max|x|（仅 CompT==float）：分块加载 x 后逐块
// SpmvVecMaxAbs 折叠；bufXWin/bufXRaw/bufProd/bufReduceWk 此时均未启用（死区）
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline float SpmvKernel<CompT, ValT, OutT>::ScanMaxAbsX() {
    AscendC::LocalTensor<float> redWk = bufReduceWk.Get<float>();
    float m = 0.0f;
    if constexpr (std::is_same_v<ValT, CompT>) {
        AscendC::LocalTensor<CompT> xWin = bufXWin.Get<CompT>();
        for (uint32_t off = 0; off < this->totalColsNum; off += XWIN_ELEMS) {
            uint32_t n = this->totalColsNum - off < XWIN_ELEMS ? this->totalColsNum - off : XWIN_ELEMS;
            AscendC::DataCopyExtParams cp{1, (uint32_t)(n * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(xWin, xVecGm[off], cp, pad);
            // MTE2→矢量归约：同上，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            // 归约 tmp 的实测安全上界为 4096（同长行段 ReduceSum），分块折叠
            for (uint32_t sub = 0; sub < n; sub += SPMV_TILE_ELEMS) {
                uint32_t sn = n - sub < SPMV_TILE_ELEMS ? n - sub : SPMV_TILE_ELEMS;
                float b = SpmvVecMaxAbs(xWin[sub], static_cast<int32_t>(sn), redWk);
                m = b > m ? b : m;
            }
            // 矢量读 xWin 毕（下一块 MTE2 覆写 xWin 的 WAR）
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    else {
        // 异类型 x：raw 分块加载转换到 bufXWin（扫描期间未启用，为死区；不能用
        // bufProd——int8->float 两步 Cast 的 tmpHalf 复用 bufProd，同址转换会
        // 踩坏未读数据）后归约
        AscendC::LocalTensor<ValT> xRaw = bufXRaw.Get<ValT>();
        AscendC::LocalTensor<CompT> xC = bufXWin.Get<CompT>();
        for (uint32_t off = 0; off < this->totalColsNum; off += XRAW_CHUNK) {
            uint32_t n = this->totalColsNum - off < XRAW_CHUNK ? this->totalColsNum - off : XRAW_CHUNK;
            AscendC::DataCopyExtParams cp{1, (uint32_t)(n * sizeof(ValT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(xRaw, xVecGm[off], cp, pad);
            // MTE2→矢量 ConvertVals：同上，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            ConvertVals(xRaw, xC, n);
            // 矢量写 xC → 矢量归约读；矢量读 xRaw 毕（下一块 MTE2 覆写 WAR）
            AscendC::PipeBarrier<PIPE_V>();
            float b = SpmvVecMaxAbs(xC, static_cast<int32_t>(n), redWk);
            m = b > m ? b : m;
        }
    }
    return m;
}

// 行组初始化：rowPtr 批量入 UB + 累加器（含 Dot2 补偿累加器）清零
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::BeginGroup(
    uint32_t groupStart, uint32_t groupCount) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    SpmvLoadGroupRowPtr(csrRowPtrGm, rowPtr, groupStart, groupCount);
    AscendC::LocalTensor<CompT> acc = bufAcc.Get<CompT>();
    AscendC::Duplicate(acc, static_cast<CompT>(0), groupCount);
    if constexpr (!std::is_same_v<CompT, int32_t>) {
        AscendC::LocalTensor<CompT> accLo = bufAccLo.Get<CompT>();
        AscendC::Duplicate(accLo, static_cast<CompT>(0), groupCount);
    }
    // 矢量清零 acc/accLo → 后续标量读改写（行段累加）；中间隔着 LoadNzTile 的全屏障，PIPE_V 已足够
    AscendC::PipeBarrier<PIPE_V>();
}

// 加载一拍 colIdx / values tile 并转换
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::LoadNzTile(int32_t t0, int32_t valid) {
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<CompT> valsC = bufValsC.Get<CompT>();
    AscendC::LocalTensor<ValT> valsRaw = bufValsRaw.Get<ValT>();
    SpmvLoadNzTile<CompT, ValT, true>(csrColIndGm, csrValGm, colIdx, valsC, valsRaw,
                                      bufProd.Get<half>(), t0, this->startValIdx,
                                      this->indexBase, valid);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::FinalizeGroup(uint32_t groupStart, uint32_t groupCount) {
    // 组收尾全屏障：去屏障实测破环（float N 路径），保留
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LocalTensor<CompT> acc = bufAcc.Get<CompT>();

    if constexpr (std::is_same_v<CompT, int32_t>) {
        // INT32 精确路径：标量整数运算
        AscendC::LocalTensor<OutT> yRaw = bufYRaw.Get<OutT>();
        if (this->readY != 0) {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(groupCount * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(yRaw, yVecGm[groupStart], cp, pad);
            // MTE2→标量读 yRaw
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        for (uint32_t r = 0; r < groupCount; r++) {
            int32_t out = this->alpha * acc.GetValue(r);
            if (this->readY != 0) {
                out += this->beta * yRaw.GetValue(r);
            }
            acc.SetValue(r, out);
        }
        // 标量写 acc→MTE3 写出：无细粒度屏障可用（PipeBarrier<PIPE_S> 实测有毒），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(groupCount * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[groupStart], acc, cp);
        // MTE3 读 acc→次组矢量清零覆写 WAR：MTE3 细粒度屏障对 V 不可靠风险，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // out = alpha * (acc + accLo) + beta * y：标量 double-single 结合（Dot2 补偿对
    // 直达输出），仅最后一次 f32 舍入；16bit 输出再经矢量 Cast 舍入。
    AscendC::LocalTensor<CompT> accLo = bufAccLo.Get<CompT>();
    __ubuf__ CompT *accPtrW = reinterpret_cast<__ubuf__ CompT *>(acc.GetPhyAddr());
    __ubuf__ CompT *accLoPtr = reinterpret_cast<__ubuf__ CompT *>(accLo.GetPhyAddr());
    __ubuf__ OutT *yRawPtr = nullptr;
    __ubuf__ CompT *yCPtr = nullptr;
    if (this->readY != 0) {
        AscendC::LocalTensor<OutT> yRaw = bufYRaw.Get<OutT>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(groupCount * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<OutT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(yRaw, yVecGm[groupStart], cp, pad);
        // MTE2→标量/矢量消费 yRaw：细粒度化实测破环（float N 路径），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same_v<CompT, OutT>) {
            yRawPtr = reinterpret_cast<__ubuf__ OutT *>(yRaw.GetPhyAddr());
        }
        else {
            AscendC::LocalTensor<CompT> yC = bufYC.Get<CompT>();
            AscendC::Cast<CompT, OutT>(yC, yRaw, AscendC::RoundMode::CAST_NONE, groupCount);
            AscendC::PipeBarrier<PIPE_V>();
            yCPtr = reinterpret_cast<__ubuf__ CompT *>(yC.GetPhyAddr());
        }
    }
    for (uint32_t r = 0; r < groupCount; r++) {
        float pA, eA;
        SpmvTwoProd(this->alpha, accPtrW[r], pA, eA);
        float resH = pA;
        float resL = eA + this->alpha * accLoPtr[r]; // alpha*accLo 的舍入为 ~2^-48 量级，可忽略
        if (this->readY != 0) {
            float yv;
            if constexpr (std::is_same_v<CompT, OutT>) {
                yv = yRawPtr[r];
            }
            else {
                yv = yCPtr[r];
            }
            float pB, eB;
            SpmvTwoProd(this->beta, yv, pB, eB);
            float t = resH + pB;
            float bp = t - resH;
            resL += (resH - (t - bp)) + (pB - bp) + eB;
            resH = t;
        }
        accPtrW[r] = static_cast<CompT>(resH + resL);
    }
    // 标量写 acc→矢量 Cast/MTE3 写出：无细粒度屏障可用，保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();

    if constexpr (std::is_same_v<CompT, OutT>) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(groupCount * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[groupStart], acc, cp);
    }
    else {
        AscendC::LocalTensor<OutT> yOut = bufYOut.Get<OutT>();
        AscendC::Cast<OutT, CompT>(yOut, acc, AscendC::RoundMode::CAST_ROUND, groupCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(groupCount * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[groupStart], yOut, cp);
    }
    // MTE3 读出→次组覆写 WAR（含矢量清零），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
}

// 一拍 tile 的行段累加：浮点路径 Pass 1 短行段 Dot2（须在矢量 Gather 改写 colIdx
// 之前执行）+ 矢量 Gather 聚集积 + Pass 2 长行段树形归约；int32 路径标量整数乘加
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::AccumulateTile(
    uint32_t &row, int32_t t0, int32_t t1, int32_t valid, uint32_t groupCount,
    uint32_t winStart, uint32_t winEnd, uint32_t winElems, uint32_t winCountTotal) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<CompT> valsC = bufValsC.Get<CompT>();
    AscendC::LocalTensor<CompT> acc = bufAcc.Get<CompT>();
    AscendC::LocalTensor<CompT> xWin = bufXWin.Get<CompT>();
    __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
    __ubuf__ int32_t *colPtr = reinterpret_cast<__ubuf__ int32_t *>(colIdx.GetPhyAddr());
    __ubuf__ CompT *accPtr = reinterpret_cast<__ubuf__ CompT *>(acc.GetPhyAddr());

    // 跳过本 tile 之前（含空行）的行
    while (row < groupCount &&
           rpPtr[row + 1] - static_cast<int32_t>(this->indexBase) <= t0) {
        row++;
    }
    if constexpr (!std::is_same_v<CompT, int32_t>) {
        // 浮点路径：短行段标量 Dot2 补偿累加（直接读 valsC/xWin，须在矢量
        // Gather 改写 colIdx 之前执行）；长行段用矢量 Gather+Mul 聚集积后
        // ReduceSum 树形归约。两段的合并均走 TwoSum 进 (acc, accLo)。
        AscendC::LocalTensor<CompT> accLo = bufAccLo.Get<CompT>();
        __ubuf__ CompT *xPtr = reinterpret_cast<__ubuf__ CompT *>(xWin.GetPhyAddr());
        __ubuf__ CompT *valPtr = reinterpret_cast<__ubuf__ CompT *>(valsC.GetPhyAddr());
        __ubuf__ CompT *accLoPtr = reinterpret_cast<__ubuf__ CompT *>(accLo.GetPhyAddr());
        // TwoProd 守卫外提谓词（行级阈值 rowSafeMax + tile 级 max|val| 归约，
        // bufReduceWk 在 Gather/ReduceSum 前为死区）；不超阈值则本 tile 短行
        // 全部走无守卫快路径，否则逐短行扫描判定
        float rowSafeMax;
        const bool tileSafe = SpmvTileSafeN(valsC, valid, this->maxAbsX,
                                            bufReduceWk.Get<float>(), rowSafeMax);
        // Pass 1：短行段标量 Dot2（Dekker 精确积 + Knuth TwoSum）
        if (winCountTotal == 1) {
            Dot2ShortRows<false>(rpPtr, colPtr, valPtr, xPtr, accPtr, accLoPtr,
                                 t0, t1, row, groupCount, winStart, winEnd,
                                 rowSafeMax, tileSafe);
        }
        else {
            // 多窗口：仅累加落在当前窗口内的非零元
            Dot2ShortRows<true>(rpPtr, colPtr, valPtr, xPtr, accPtr, accLoPtr,
                                t0, t1, row, groupCount, winStart, winEnd,
                                rowSafeMax, tileSafe);
        }
        // 矢量 Gather 聚集 x（列索引在本 tile 之后不再使用）
        AscendC::LocalTensor<CompT> prod = bufProd.Get<CompT>();
        if (winCountTotal == 1) {
            GatherXWindow<true>(prod, colIdx, xWin, valid, winStart, winElems);
        }
        else {
            GatherXWindow<false>(prod, colIdx, xWin, valid, winStart, winElems);
        }
        // Gather 的写在 arch22 上不被 PipeBarrier<PIPE_V> 覆盖（实测 36 例确定性失败），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(prod, prod, valsC, valid);
        // 矢量写 prod → Pass 2 标量头段读/矢量 ReduceSum（ReduceProdSegment）：同上，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        // Pass 2：长行段（>= SPMV_LONG_ROW_MIN）走矢量积 + 树形归约，
        // 短行段已在 Pass 1 用 Dot2 处理
        ReduceLongRows(rpPtr, accPtr, accLoPtr, prod, t0, t1, row, groupCount);
    }
    else {
        // INT32 精确路径：标量整数乘加，与归约顺序无关（exact match）
        AccumulateTileInt32(rpPtr, colPtr, valsC, xWin, accPtr,
                            t0, t1, row, groupCount, winStart, winEnd);
    }
}

template <typename CompT, typename ValT, typename OutT>
template <bool MULTI_WIN>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Dot2ShortRows(
    const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
    const __ubuf__ float *valPtr, __ubuf__ float *xPtr,
    __ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
    int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount,
    uint32_t winStart, uint32_t winEnd, float rowSafeMax, bool tileSafe) {
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, this->indexBase, row, groupCount, t0, t1);
    uint32_t r1;
    while (cur.Next(r1)) {
        int32_t len = cur.hi - cur.lo;
        if (len < static_cast<int32_t>(SPMV_LONG_ROW_MIN)) {
            bool safe = tileSafe;
            if (!safe) {
                // 行级预扫描：行内（限当前窗口）max|val| <= rowSafeMax 则该行
                // 仍走无守卫快路径（谓词保守：上界乘积不越限 ⇒ 每元素都安全）
                float rowMax = 0.0f;
                for (int32_t j = cur.lo; j < cur.hi; j++) {
                    int32_t col = colPtr[j - t0];
                    if constexpr (MULTI_WIN) {
                        if (static_cast<uint32_t>(col) < winStart ||
                            static_cast<uint32_t>(col) >= winEnd)
                            continue;
                    }
                    float av = SpmvAbsF(valPtr[j - t0]);
                    rowMax = av > rowMax ? av : rowMax;
                }
                safe = rowMax <= rowSafeMax;
            }
            float s = 0.0f, c = 0.0f;
            if (safe) {
                SpmvDot2RowSeg<MULTI_WIN, false>(cur.lo, cur.hi, t0, colPtr, valPtr, xPtr,
                                                 winStart, winEnd, s, c);
            }
            else {
                SpmvDot2RowSeg<MULTI_WIN, true>(cur.lo, cur.hi, t0, colPtr, valPtr, xPtr,
                                                winStart, winEnd, s, c);
            }
            float a = accPtr[r1];
            float t = a + s;
            float bp = t - a;
            accLoPtr[r1] += ((a - (t - bp)) + (s - bp)) + c;
            accPtr[r1] = t;
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
template <bool SINGLE_WIN>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::GatherXWindow(
    AscendC::LocalTensor<CompT> prod,
    const AscendC::LocalTensor<int32_t> &colIdx,
    const AscendC::LocalTensor<CompT> &xWin,
    int32_t valid, uint32_t winStart, uint32_t winElems) {
    if constexpr (SINGLE_WIN) {
        // 单窗口：矢量 Gather 聚集 x（列索引天然落在 [0, cols) 内，无需钳位）。
        // vgather 的偏移以字节为单位，colIdx 先乘 sizeof(CompT)（本 tile 之后不再使用）
        AscendC::Muls(colIdx, colIdx, static_cast<int32_t>(sizeof(CompT)), valid);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<uint32_t> offs = colIdx.ReinterpretCast<uint32_t>();
        AscendC::Gather(prod, xWin, offs, 0, static_cast<uint32_t>(valid));
    }
    else {
        // 多窗口：t = c - winStart 钳位到 [0, winElems-1] 后矢量 Gather；
        // 出窗元素由「钳位前后相等」掩码经 Select 置 0
        AscendC::LocalTensor<int32_t> prodI32 = prod.template ReinterpretCast<int32_t>();
        AscendC::Adds(colIdx, colIdx, -static_cast<int32_t>(winStart), valid);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(prodI32, colIdx, 0, valid); // 备份钳位前偏移
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(colIdx, colIdx, 0, valid);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(colIdx, colIdx, static_cast<int32_t>(winElems) - 1, valid);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<uint8_t> inWin = bufReduceWk.Get<uint8_t>();
        AscendC::Compare(inWin, prodI32, colIdx, AscendC::CMPMODE::EQ,
                         static_cast<uint32_t>(valid));
        AscendC::Muls(colIdx, colIdx, static_cast<int32_t>(sizeof(CompT)), valid);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<uint32_t> offs = colIdx.ReinterpretCast<uint32_t>();
        AscendC::Gather(prod, xWin, offs, 0, static_cast<uint32_t>(valid));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(prod, inWin, prod, static_cast<CompT>(0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        static_cast<uint32_t>(valid));
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::ReduceLongRows(
    const __ubuf__ int32_t *rpPtr, __ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
    const AscendC::LocalTensor<CompT> &prod,
    int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount) {
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, this->indexBase, row, groupCount, t0, t1);
    uint32_t r;
    while (cur.Next(r)) {
        int32_t len = cur.hi - cur.lo;
        if (len >= static_cast<int32_t>(SPMV_LONG_ROW_MIN)) {
            float accLocal = ReduceProdSegment(prod, cur.lo - t0, len);
            float a = accPtr[r];
            float t = a + accLocal;
            float bp = t - a;
            accLoPtr[r] += (a - (t - bp)) + (accLocal - bp);
            accPtr[r] = t;
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::AccumulateTileInt32(
    const __ubuf__ int32_t *rpPtr, const __ubuf__ int32_t *colPtr,
    const AscendC::LocalTensor<CompT> &valsC,
    const AscendC::LocalTensor<CompT> &xWin,
    __ubuf__ CompT *accPtr, int32_t t0, int32_t t1,
    uint32_t row, uint32_t groupCount,
    uint32_t winStart, uint32_t winEnd) {
    __ubuf__ CompT *valPtr = reinterpret_cast<__ubuf__ CompT *>(valsC.GetPhyAddr());
    __ubuf__ CompT *xPtr = reinterpret_cast<__ubuf__ CompT *>(xWin.GetPhyAddr());
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, this->indexBase, row, groupCount, t0, t1);
    uint32_t r;
    while (cur.Next(r)) {
        CompT accLocal = static_cast<CompT>(0);
        for (int32_t j = cur.lo; j < cur.hi; j++) {
            int32_t c = colPtr[j - t0];
            if (static_cast<uint32_t>(c) >= winStart && static_cast<uint32_t>(c) < winEnd) {
                accLocal += xPtr[c - static_cast<int32_t>(winStart)] * valPtr[j - t0];
            }
        }
        accPtr[r] = accPtr[r] + accLocal;
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Process() {
    if (this->blockRowNum == 0)
        return;

    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();

    const uint32_t winCountTotal =
        (this->totalColsNum + XWIN_ELEMS - 1) / XWIN_ELEMS;

    if constexpr (!std::is_same_v<CompT, int32_t>) {
        // TwoProd 守卫外提：一次性归约全向量 max|x|（x 内容跨行组/窗口不变）
        this->maxAbsX = ScanMaxAbsX();
    }

    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        uint32_t groupCount = this->blockRowNum - groupStart;
        if (groupCount > SPMV_ROW_GROUP)
            groupCount = SPMV_ROW_GROUP;

        BeginGroup(groupStart, groupCount);
        int32_t groupNnz0, groupNnz1;
        SpmvGroupNnzRange(rowPtr, groupCount, this->indexBase, groupNnz0, groupNnz1);

        for (uint32_t w = 0; w < winCountTotal; w++) {
            const uint32_t winStart = w * XWIN_ELEMS;
            uint32_t winElems = this->totalColsNum - winStart;
            if (winElems > XWIN_ELEMS)
                winElems = XWIN_ELEMS;
            const uint32_t winEnd = winStart + winElems;
            LoadXWindow(winStart, winElems);

            // (行, tile) 双指针走查本组 nnz 区间
            uint32_t row = 0;
            for (int32_t t0 = groupNnz0; t0 < groupNnz1; t0 += SPMV_TILE_ELEMS) {
                int32_t valid;
                const int32_t t1 = SpmvBeginTile(t0, groupNnz1, valid);
                LoadNzTile(t0, valid);
                AccumulateTile(row, t0, t1, valid, groupCount,
                               winStart, winEnd, winElems, winCountTotal);
            }
        }
        FinalizeGroup(groupStart, groupCount);
    }
}

/**
 * @brief CSR 格式稀疏矩阵转置-稠密向量乘法（SpMV Transpose）AscendC 算子
 *
 * 行并行遍历非零元，贡献先聚合到 UB 中的 y 累加窗口（按列分窗，窗口内 UB 累加），
 * 窗口结束后合并到 GM。多遍窗口之间复用同一 UB 缓冲。
 * beta 缩放阶段按列均分，多核不重叠；readY == 0 时不读 y。
 * INT32 路径采用整数原子加，保证 exact match。
 * f32 计算路径（DS_WS，输出 f32/f16/bf16）：GM 原子加与 beta 预缩放的
 * 逐级 f32 舍入在正负消减（golden 近零）场景下超出 ATK 严格判定
 * （rtol=1e-5, atol=1e-8，验收扩充用例 gen0751 实测）；16bit 输出虽单次
 * 舍入即 1 ulp，但原子加顺序非确定会在舍入边界上间歇翻转。统一改用确定性
 * double-single 方案：核内窗口累加用 Dekker 精确积 + Knuth TwoSum 补偿
 * （acc, accLo），每核把 (acc, accLo) 普通写入自己专属的 ws 平面（不用
 * 原子加，跨核顺序无关），最后按列切分做 finalize：跨核 TwoSum 链合并 +
 * beta*y0 的 double-single 结合，仅一次输出舍入（16bit 输出在写出时
 * 由 f32 转 half/bf16）。ws 布局：groups 个 hi 平面 + groups 个 lo 平面，
 * 平面内按 32B 对齐的行 stride 排列，groups = min(blockNum, max(rows, 1))。
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernelTrans {
public:
    // f32 计算（f32/f16/bf16 输出）使用 double-single workspace 路径
    //（确定性，无原子加舍入）；int32 路径仍用整数原子加
    static constexpr bool DS_WS = std::is_same_v<CompT, float>;
    // y 累加窗口元素数（fp32/int32 均为 4B）；DS_WS 需额外 accLo 窗口，窗口减半
    static constexpr uint32_t YWIN_ELEMS =
        (DS_WS ? SPMV_XWIN_BYTES / 2 : SPMV_XWIN_BYTES) / sizeof(float) / 256 * 256;

    __aicore__ inline SpmvKernelTrans(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        GM_ADDR ws,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        CompT alpha, CompT beta, uint32_t indexBase, uint32_t readY);
    __aicore__ inline void Process();
    // 单行段贡献累加（游标已保证 lo < hi）：非 DS 路径直接乘加进 yAcc 窗口
    //（public：供文件级共用模板 SpmvAccumulateTileSegs(Ds) 回调）
    __aicore__ inline void AccumulateSeg(uint32_t r, int32_t lo, int32_t hi,
                                         const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr,
                                         const __ubuf__ CompT *xRowPtr, __ubuf__ CompT *yAccPtr,
                                         int32_t t0, uint32_t winStart, uint32_t winEnd);
    // DS_WS 单行段：贡献 alpha*val*x 用两级 Dekker 精确积、Knuth TwoSum
    // 补偿累加进 (acc, accLo) 双字窗口。GUARDED=false 为无守卫快路径
    //（调用方谓词已保证本行全部乘积安全），快慢路径对正常输入逐位一致
    template <bool GUARDED>
    __aicore__ inline void AccumulateSegDs(uint32_t r, int32_t lo, int32_t hi,
                                           const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr,
                                           const __ubuf__ CompT *xRowPtr, __ubuf__ float *accPtr,
                                           __ubuf__ float *accLoPtr, int32_t t0,
                                           uint32_t winStart, uint32_t winEnd);
    // DS_WS 行级安全谓词（守卫外提）：行段内（限窗口）max|val| 与本行 |x| 的
    // 上界乘积覆盖两级 Dekker 精确积（val*x 与 alpha*p1）的全部溢出风险
    __aicore__ inline bool RowDsSafe(uint32_t r, int32_t lo, int32_t hi,
                                     const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr,
                                     const __ubuf__ CompT *xRowPtr, int32_t t0,
                                     uint32_t winStart, uint32_t winEnd, float absAlphaMax);

private:
    __aicore__ inline void ScaleBeta();
    __aicore__ inline void ConvertVals(const AscendC::LocalTensor<ValT> &raw,
                                       const AscendC::LocalTensor<CompT> &dst, uint32_t count);
    __aicore__ inline void LoadXGroup(uint32_t globalRow, uint32_t count);
    // 行组头：rowPtr 批量入 UB + 本组 x 入 UB + 本组 nnz 区间
    __aicore__ inline SpmvGroupHead BeginGroup(uint32_t groupStart);
    __aicore__ inline void AccumulateWindow(uint32_t winStart, uint32_t winEnd);
    __aicore__ inline void AccumulateWindowDs(uint32_t winStart, uint32_t winEnd);
    __aicore__ inline void FinalizeDs();
    // 矢量 TwoSum 合并第 g 组 (hi, lo) 平面：(accHi,accLo) += (stgHi,stgLo)
    __aicore__ inline void MergeDsGroup(uint32_t g, uint32_t chunkSize, uint32_t col0);
    // beta*y0 的 double-single 结合（标量）：f32 输出单次舍入直写 outPtr；
    // 16bit 输出 s/lo 回写 hiPtr/loPtr 供中点重判
    __aicore__ inline void CombineBetaY0(uint32_t chunkSize, __ubuf__ float *hiPtr,
                                         __ubuf__ float *loPtr, const __ubuf__ float *y0Ptr,
                                         __ubuf__ float *outPtr);
    // 16bit 输出的 RNE 中点重判（见 SpmvRnePickNeighbors/SpmvRneMidpointRecheck）
    __aicore__ inline void RneMidpointFix16(uint32_t chunkSize, const __ubuf__ float *hiPtr,
                                            const __ubuf__ float *loPtr, const __ubuf__ float *stgPtr);

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRowPtr, bufXRow, bufXRowRaw;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufColIdx, bufValsC, bufValsRaw;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYAcc, bufYAccLo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYRaw, bufYC;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<ValT> csrValGm;
    AscendC::GlobalTensor<ValT> xVecGm;
    AscendC::GlobalTensor<OutT> yVecGm;
    AscendC::GlobalTensor<float> wsGm; // 16bit 输出：fp32 原子累加 ws；f32 输出：double-single 平面

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t startValIdx;
    uint32_t endValIdx;
    CompT alpha;
    CompT beta;
    uint32_t indexBase;
    uint32_t readY;

    uint32_t yBlockStart;
    uint32_t yBlockCount;
    uint32_t dsGroups;   // DS_WS：参与累加的核组数 = min(blockNum, max(rows,1))
    uint32_t dsStride;   // DS_WS：每核每平面的 float 数（32B 对齐）
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    GM_ADDR ws,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    CompT alpha, CompT beta, uint32_t indexBase, uint32_t readY) {
    SpmvInitCommon(totalRowsNum, totalColsNum, indexBase, readY,
                   this->totalRowsNum, this->totalColsNum, this->indexBase, this->readY);
    this->alpha = alpha;
    this->beta = beta;

    uint32_t blockNum = AscendC::GetBlockNum();
    if constexpr (DS_WS) {
        uint32_t rr = totalRowsNum > 0 ? totalRowsNum : 1;
        this->dsGroups = blockNum > 0 && blockNum < rr ? blockNum : rr;
        // 每核每平面 float 数按 32B（8 float）对齐，保证各组平面基址 32B 对齐
        this->dsStride = (totalColsNum + 7) / 8 * 8;
    }
    if (blockNum > 0) {
        // 均分行：前 tempRowNum 个核各多拿一行
        SpmvSplitBlockRange(totalRowsNum, blockNum, this->blockRowNum, this->startRow);
        // beta 缩放按列均分
        SpmvSplitBlockRange(totalColsNum, blockNum, this->yBlockCount, this->yBlockStart);
    }
    else {
        this->blockRowNum = 0;
        this->yBlockCount = 0;
    }

    xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalRowsNum);
    yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec, this->totalColsNum);
    if constexpr (DS_WS) {
        wsGm.SetGlobalBuffer((__gm__ float *)ws, 2 * this->dsGroups * this->dsStride);
    }

    SpmvBindRowRange(csrRowPtrGm, csrRowPtr, this->startRow, this->blockRowNum, this->indexBase,
                     this->startValIdx, this->endValIdx);
    if (this->blockRowNum > 0) {
        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + this->startValIdx,
                                    this->endValIdx - this->startValIdx);
        csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + this->startValIdx,
                                 this->endValIdx - this->startValIdx);
    }

    pipe.InitBuffer(bufRowPtr, (SPMV_ROW_GROUP + 1) * sizeof(int32_t));
    pipe.InitBuffer(bufXRow, SPMV_ROW_GROUP * sizeof(CompT));
    pipe.InitBuffer(bufXRowRaw,
        std::is_same_v<ValT, CompT> ? 32 : SPMV_ROW_GROUP * sizeof(ValT));
    pipe.InitBuffer(bufColIdx, SPMV_TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufValsC, SPMV_TILE_ELEMS * sizeof(CompT));
    pipe.InitBuffer(bufValsRaw,
        std::is_same_v<ValT, CompT> ? 32 : SPMV_TILE_ELEMS * sizeof(ValT));
    pipe.InitBuffer(bufYAcc, YWIN_ELEMS * sizeof(CompT));
    if constexpr (DS_WS) {
        pipe.InitBuffer(bufYAccLo, YWIN_ELEMS * sizeof(float));
    }
    pipe.InitBuffer(bufYRaw, SPMV_ROW_GROUP * sizeof(OutT));
    pipe.InitBuffer(bufYC, SPMV_ROW_GROUP * sizeof(CompT));
}

// ValT -> CompT 转换（委托共用原语；int8 走标量转换，本类无积缓冲可复用）
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ConvertVals(
    const AscendC::LocalTensor<ValT> &raw, const AscendC::LocalTensor<CompT> &dst, uint32_t count) {
    SpmvConvertVals<CompT, ValT, false>(raw, dst, count, bufYC.Get<half>());
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ScaleBeta() {
    if (this->yBlockCount == 0)
        return;
    // beta == 1 且需要保留 y：无需缩放
    if (this->readY != 0 && this->beta == static_cast<CompT>(1))
        return;

    AscendC::LocalTensor<OutT> yChunk = bufYRaw.Get<OutT>();
    AscendC::LocalTensor<CompT> yC = bufYC.Get<CompT>();
    for (uint32_t offset = 0; offset < this->yBlockCount; offset += SPMV_ROW_GROUP) {
        uint32_t chunkSize = SPMV_ROW_GROUP;
        if (offset + chunkSize > this->yBlockCount)
            chunkSize = this->yBlockCount - offset;

        if (this->readY == 0) {
            // beta == 0：不读 y，直接写 0（16bit 输出先置 CompT 0 再矢量 Cast，避免标量 bf16 转换）
            if constexpr (std::is_same_v<CompT, OutT>) {
                AscendC::Duplicate(yChunk, static_cast<OutT>(0), chunkSize);
            }
            else {
                AscendC::Duplicate(yC, static_cast<CompT>(0), chunkSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<OutT, CompT>(yChunk, yC, AscendC::RoundMode::CAST_ROUND, chunkSize);
            }
            // 矢量写 yChunk → MTE3 写回
            AscendC::PipeBarrier<PIPE_V>();
        }
        else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunk, yVecGm[this->yBlockStart + offset], cp, padParams);
            if constexpr (std::is_same_v<CompT, int32_t>) {
                // MTE2→标量读 yChunk
                AscendC::PipeBarrier<PIPE_MTE2>();
                // 整数路径标量缩放，保持精确
                for (uint32_t i = 0; i < chunkSize; i++)
                    yChunk.SetValue(i, yChunk.GetValue(i) * static_cast<OutT>(this->beta));
                // 标量写 yChunk→MTE3 写回：无细粒度屏障可用（PipeBarrier<PIPE_S> 实测有毒），保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            else if constexpr (std::is_same_v<CompT, OutT>) {
                // MTE2→矢量 Muls：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Muls(yChunk, yChunk, static_cast<OutT>(this->beta), chunkSize);
                // 矢量写 yChunk → MTE3 写回
                AscendC::PipeBarrier<PIPE_V>();
            }
            else {
                // MTE2→矢量 Cast：同上，保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast<CompT, OutT>(yC, yChunk, AscendC::RoundMode::CAST_NONE, chunkSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(yC, yC, this->beta, chunkSize);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<OutT, CompT>(yChunk, yC, AscendC::RoundMode::CAST_ROUND, chunkSize);
                // 矢量写 yChunk → MTE3 写回
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunk, cp2);
        // MTE3 读 yChunk→次 chunk 矢量/MTE2 覆写 WAR：MTE3 细粒度屏障对 V 不可靠风险，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::LoadXGroup(uint32_t globalRow, uint32_t count) {
    // xRow/xRaw 的消费者均为标量读（随程序序完成），MTE2 覆写前无需屏障
    AscendC::LocalTensor<CompT> xRow = bufXRow.Get<CompT>();
    if constexpr (std::is_same_v<ValT, CompT>) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xRow, xVecGm[globalRow], cp, pad);
        // MTE2→标量读 xRow
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    else {
        AscendC::LocalTensor<ValT> xRaw = bufXRowRaw.Get<ValT>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(count * sizeof(ValT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<ValT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xRaw, xVecGm[globalRow], cp, pad);
        // MTE2→矢量 ConvertVals：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        ConvertVals(xRaw, xRow, count);
        // 矢量写 xRow → 标量读
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// 行组头：rowPtr 批量入 UB + 本组 x 入 UB + 本组 nnz 区间
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline SpmvGroupHead SpmvKernelTrans<CompT, ValT, OutT>::BeginGroup(
    uint32_t groupStart) {
    SpmvGroupHead head;
    head.groupCount = this->blockRowNum - groupStart;
    if (head.groupCount > SPMV_ROW_GROUP)
        head.groupCount = SPMV_ROW_GROUP;
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    SpmvLoadGroupRowPtr(csrRowPtrGm, rowPtr, groupStart, head.groupCount);
    LoadXGroup(this->startRow + groupStart, head.groupCount);
    SpmvGroupNnzRange(rowPtr, head.groupCount, this->indexBase, head.groupNnz0, head.groupNnz1);
    return head;
}

// 遍历本核全部行，将落在 [winStart, winEnd) 内的贡献累加到 UB y 窗口
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::AccumulateWindow(
    uint32_t winStart, uint32_t winEnd) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<CompT> xRow = bufXRow.Get<CompT>();
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<CompT> valsC = bufValsC.Get<CompT>();
    AscendC::LocalTensor<ValT> valsRaw = bufValsRaw.Get<ValT>();
    AscendC::LocalTensor<CompT> yAcc = bufYAcc.Get<CompT>();

    uint32_t row = 0;
    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        row = 0;
        const SpmvGroupHead head = BeginGroup(groupStart);
        const uint32_t groupCount = head.groupCount;
        for (int32_t t0 = head.groupNnz0; t0 < head.groupNnz1; t0 += SPMV_TILE_ELEMS) {
            int32_t valid;
            const int32_t t1 = SpmvBeginTile(t0, head.groupNnz1, valid);
            SpmvLoadNzTile<CompT, ValT, false>(csrColIndGm, csrValGm, colIdx, valsC, valsRaw,
                                               bufYC.Get<half>(), t0, this->startValIdx,
                                               this->indexBase, valid);

            __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
            __ubuf__ int32_t *colPtr = reinterpret_cast<__ubuf__ int32_t *>(colIdx.GetPhyAddr());
            __ubuf__ CompT *valPtr = reinterpret_cast<__ubuf__ CompT *>(valsC.GetPhyAddr());
            __ubuf__ CompT *xRowPtr = reinterpret_cast<__ubuf__ CompT *>(xRow.GetPhyAddr());
            __ubuf__ CompT *yAccPtr = reinterpret_cast<__ubuf__ CompT *>(yAcc.GetPhyAddr());
            SpmvAccumulateTileSegs(*this, this->indexBase, rpPtr, colPtr, valPtr, xRowPtr,
                                   yAccPtr, row, groupCount, t0, t1, winStart, winEnd);
        }
    }
}

// DS_WS：遍历本核全部行，落在 [winStart, winEnd) 内的贡献以 Dot2 补偿累加进
// UB (acc, accLo) 双字窗口（alpha*val*x 用 Dekker 精确积，累加用 Knuth TwoSum）
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::AccumulateWindowDs(
    uint32_t winStart, uint32_t winEnd) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<CompT> xRow = bufXRow.Get<CompT>();
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<CompT> valsC = bufValsC.Get<CompT>();
    AscendC::LocalTensor<ValT> valsRaw = bufValsRaw.Get<ValT>();
    AscendC::LocalTensor<CompT> yAcc = bufYAcc.Get<CompT>();
    AscendC::LocalTensor<float> yAccLo = bufYAccLo.Get<float>();
    const float absAlpha = SpmvAbsF(static_cast<float>(this->alpha));

    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        const SpmvGroupHead head = BeginGroup(groupStart);
        const uint32_t groupCount = head.groupCount;
        // 守卫外提谓词的组级量：本组 x 的 max|x|（矢量归约，bufYC 累加阶段为死区）。
        // 同类型档 xRow 由 DMA 直写（LoadXGroup 仅 PIPE_MTE2 收尾，MTE2→V 实测
        // 不可靠），归约前补全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        const float grpMaxX = SpmvVecMaxAbs(xRow, static_cast<int32_t>(groupCount),
                                            bufYC.Get<float>());
        uint32_t row = 0;
        for (int32_t t0 = head.groupNnz0; t0 < head.groupNnz1; t0 += SPMV_TILE_ELEMS) {
            int32_t valid;
            const int32_t t1 = SpmvBeginTile(t0, head.groupNnz1, valid);
            SpmvLoadNzTile<CompT, ValT, false>(csrColIndGm, csrValGm, colIdx, valsC, valsRaw,
                                               bufYC.Get<half>(), t0, this->startValIdx,
                                               this->indexBase, valid);

            __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
            // DS 双字窗口基址（与 rowPtr 紧邻取址）
            __ubuf__ float *accPtr = reinterpret_cast<__ubuf__ float *>(yAcc.GetPhyAddr());
            __ubuf__ float *accLoPtr = reinterpret_cast<__ubuf__ float *>(yAccLo.GetPhyAddr());
            __ubuf__ int32_t *colPtr = reinterpret_cast<__ubuf__ int32_t *>(colIdx.GetPhyAddr());
            __ubuf__ CompT *valPtr = reinterpret_cast<__ubuf__ CompT *>(valsC.GetPhyAddr());
            __ubuf__ CompT *xRowPtr = reinterpret_cast<__ubuf__ CompT *>(xRow.GetPhyAddr());
            // tile 级谓词（见 SpmvTileSafeT；bufYC 累加阶段为归约死区）
            const bool tileSafe = SpmvTileSafeT(valsC, valid, grpMaxX, absAlpha,
                                                bufYC.Get<float>());
            SpmvAccumulateTileSegsDs(*this, this->indexBase, rpPtr, colPtr, valPtr, xRowPtr,
                                     accPtr, accLoPtr, row, groupCount, t0, t1, winStart, winEnd,
                                     tileSafe, absAlpha);
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::AccumulateSeg(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr, const __ubuf__ CompT *xRowPtr,
    __ubuf__ CompT *yAccPtr, int32_t t0, uint32_t winStart, uint32_t winEnd) {
    CompT xVal = xRowPtr[r];
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        if (static_cast<uint32_t>(c) >= winStart && static_cast<uint32_t>(c) < winEnd) {
            uint32_t slot = static_cast<uint32_t>(c) - winStart;
            yAccPtr[slot] = yAccPtr[slot] + this->alpha * valPtr[j - t0] * xVal;
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
template <bool GUARDED>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::AccumulateSegDs(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr, const __ubuf__ CompT *xRowPtr,
    __ubuf__ float *accPtr, __ubuf__ float *accLoPtr, int32_t t0,
    uint32_t winStart, uint32_t winEnd) {
    float xVal = static_cast<float>(xRowPtr[r]);
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        if (static_cast<uint32_t>(c) >= winStart && static_cast<uint32_t>(c) < winEnd) {
            uint32_t slot = static_cast<uint32_t>(c) - winStart;
            // 贡献 = alpha * val * x，两级 Dekker 精确积
            float p1, e1, p2, e2;
            if constexpr (GUARDED) {
                SpmvTwoProd(static_cast<float>(valPtr[j - t0]), xVal, p1, e1);
                SpmvTwoProd(this->alpha, p1, p2, e2);
            }
            else {
                SpmvTwoProdFast(static_cast<float>(valPtr[j - t0]), xVal, p1, e1);
                SpmvTwoProdFast(this->alpha, p1, p2, e2);
            }
            float cLo = e2 + this->alpha * e1; // ~2^-48 量级，可忽略其舍入
            // TwoSum 合并入 (acc, accLo)
            float s, e;
            SpmvTwoSum(accPtr[slot], p2, s, e);
            accLoPtr[slot] += e + cLo;
            accPtr[slot] = s;
        }
    }
}

// DS_WS 行级安全谓词：rowMax = 行段内（限窗口）max|val|，xv = |x[r]|；
// pv = rowMax*xv 为 |val*x| 上界。安全 ⟺ rowMax、xv、pv 均在拆分安全域
// （4097 倍不溢出）且 pv*|alpha| 在积安全域（第二级 alpha*p1 的高位积
// 不溢出）。上界乘法自身溢出为 Inf 时判定不安全，方向正确
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline bool SpmvKernelTrans<CompT, ValT, OutT>::RowDsSafe(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ CompT *valPtr, const __ubuf__ CompT *xRowPtr,
    int32_t t0, uint32_t winStart, uint32_t winEnd, float absAlphaMax) {
    const float xv = SpmvAbsF(static_cast<float>(xRowPtr[r]));
    float rowMax = 0.0f;
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        if (static_cast<uint32_t>(c) >= winStart && static_cast<uint32_t>(c) < winEnd) {
            float av = SpmvAbsF(static_cast<float>(valPtr[j - t0]));
            rowMax = av > rowMax ? av : rowMax;
        }
    }
    const float pv = rowMax * xv;
    return rowMax <= SPMV_TP_SPLIT_SAFE_MAX && xv <= SPMV_TP_SPLIT_SAFE_MAX &&
           pv <= SPMV_TP_SPLIT_SAFE_MAX && absAlphaMax <= SPMV_TP_SPLIT_SAFE_MAX &&
           pv * absAlphaMax <= SPMV_TP_PROD_SAFE_MAX;
}

// DS_WS finalize：按列切分，跨核 TwoSum 链合并各组 (hi, lo) 平面，
// 再与 beta*y0 做 double-single 结合，单次舍入写出
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::FinalizeDs() {
    if (this->yBlockCount == 0)
        return;
    // 组间合并的 staging 复用累加窗口（累加已结束）；accHi 用 bufYC，accLo 用
    // bufYAccLo 尾部（偏移 4096，避开 staging 区）
    const uint32_t accLoOff = 4096;

    for (uint32_t offset = 0; offset < this->yBlockCount; offset += SPMV_ROW_GROUP) {
        uint32_t chunkSize = SPMV_ROW_GROUP;
        if (offset + chunkSize > this->yBlockCount)
            chunkSize = this->yBlockCount - offset;
        const uint32_t col0 = this->yBlockStart + offset;

        AscendC::LocalTensor<float> accHi = bufYC.Get<float>();
        AscendC::LocalTensor<float> accLo = bufYAccLo.Get<float>();
        AscendC::Duplicate(accHi, 0.0f, chunkSize);
        AscendC::Duplicate(accLo[accLoOff], 0.0f, chunkSize);
        // DS finalize 起点屏障：去屏障/细粒度化实测破环（trans 路径），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t g = 0; g < this->dsGroups; g++) {
            MergeDsGroup(g, chunkSize, col0);
        }

        // beta*y0 的 double-single 结合（标量），单次舍入写出。
        // f32 输出直接写 yChunk；16bit 输出：staging（tmpV）写 f32(s+lo)
        //（保留 lo 的量级，消减场景可跨多个 16bit 档），s 留在 accHi、lo 留在
        // accLo staging；先矢量 Cast（CAST_RINT，平局取偶，与验收 golden 的
        // RNE 口径一致）；再对 r 恰落 16bit 舍入中点的元素用双字重判方向
        //（d3 = (s - mid) + lo），消除双舍入在中点上的系统性误判
        AscendC::LocalTensor<float> tmpV = bufXRow.Get<float>();
        __ubuf__ float *hiPtr = reinterpret_cast<__ubuf__ float *>(accHi.GetPhyAddr());
        __ubuf__ float *loPtr = reinterpret_cast<__ubuf__ float *>(accLo.GetPhyAddr()) + accLoOff;
        AscendC::LocalTensor<OutT> yChunk = bufYRaw.Get<OutT>();
        __ubuf__ OutT *yPtr = reinterpret_cast<__ubuf__ OutT *>(yChunk.GetPhyAddr());
        __ubuf__ float *stgPtr = reinterpret_cast<__ubuf__ float *>(tmpV.GetPhyAddr());
        constexpr bool OUT_F32 = std::is_same_v<OutT, float>;
        if (this->readY != 0) {
            AscendC::DataCopyExtParams cpY{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<OutT> padY{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunk, yVecGm[col0], cpY, padY);
            // MTE2→标量/矢量消费 y0：细粒度化实测破环（trans DS_WS 路径），保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            // 16bit 输出：bisheng 后端不支持标量 bf16 类型转换，先用矢量
            // Cast（CAST_NONE，精确）把 y0 转成 f32，再走相同的标量 DS 结合
            if constexpr (OUT_F32) {
                CombineBetaY0(chunkSize, hiPtr, loPtr, yPtr, yPtr);
            }
            else {
                AscendC::LocalTensor<float> tmpT = bufColIdx.Get<float>();
                AscendC::Cast<float, OutT>(tmpT, yChunk, AscendC::RoundMode::CAST_NONE, chunkSize);
                // 矢量写 tmpT→标量读：细粒度化实测破环（trans DS_WS 路径），保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
                CombineBetaY0(chunkSize, hiPtr, loPtr,
                              reinterpret_cast<__ubuf__ float *>(tmpT.GetPhyAddr()), stgPtr);
            }
        }
        else {
            for (uint32_t j = 0; j < chunkSize; j++) {
                if constexpr (OUT_F32) {
                    yPtr[j] = hiPtr[j] + loPtr[j];
                } else {
                    stgPtr[j] = hiPtr[j] + loPtr[j];   // s 留在 hiPtr，lo 留在 loPtr
                }
            }
        }
        if constexpr (!OUT_F32) {
            RneMidpointFix16(chunkSize, hiPtr, loPtr, stgPtr);
        }
        // 标量/矢量写 yChunk→MTE3 写出（含标量写路径），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cpO{1, (uint32_t)(chunkSize * sizeof(OutT)), 0, 0, 0};
        AscendC::DataCopyPad(yVecGm[col0], yChunk, cpO);
        // MTE3 读出→次 chunk 覆写 WAR，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::MergeDsGroup(
    uint32_t g, uint32_t chunkSize, uint32_t col0) {
    AscendC::LocalTensor<float> stgHi = bufYAcc.Get<float>();
    AscendC::LocalTensor<float> stgLo = bufYAccLo.Get<float>();
    AscendC::LocalTensor<float> accHi = bufYC.Get<float>();
    AscendC::LocalTensor<float> accLo = bufYAccLo.Get<float>();
    AscendC::LocalTensor<float> tmpT = bufColIdx.Get<float>();
    AscendC::LocalTensor<float> tmpU = bufValsC.Get<float>();
    AscendC::LocalTensor<float> tmpV = bufXRow.Get<float>();
    const uint32_t accLoOff = 4096;
    uint64_t hiBase = static_cast<uint64_t>(g) * this->dsStride + col0;
    uint64_t loBase = static_cast<uint64_t>(this->dsGroups + g) * this->dsStride + col0;
    AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(stgHi, wsGm[hiBase], cp, pad);
    AscendC::DataCopyPad(stgLo, wsGm[loBase], cp, pad);
    // MTE2（staging 入 UB）→矢量 TwoSum：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    // 矢量 TwoSum：(accHi,accLo) += (stgHi,stgLo)
    AscendC::Add(tmpT, accHi, stgHi, chunkSize);            // t = a + b
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpU, tmpT, accHi, chunkSize);             // bp = t - a
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpV, tmpT, tmpU, chunkSize);              // t - bp
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpV, accHi, tmpV, chunkSize);             // a - (t - bp)
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpU, stgHi, tmpU, chunkSize);             // b - bp
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmpV, tmpV, tmpU, chunkSize);              // e
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmpU, stgLo, accLo[accLoOff], chunkSize);  // lo 项合并（舍入可忽略）
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(accLo[accLoOff], tmpV, tmpU, chunkSize);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(accHi, tmpT, 0.0f, chunkSize);            // accHi = t（拷回）
    // DS 合并链收尾屏障：细粒度化实测破环（trans 路径），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CombineBetaY0(
    uint32_t chunkSize, __ubuf__ float *hiPtr, __ubuf__ float *loPtr,
    const __ubuf__ float *y0Ptr, __ubuf__ float *outPtr) {
    for (uint32_t j = 0; j < chunkSize; j++) {
        float pB, eB, s, e;
        float y0 = y0Ptr[j];
        SpmvTwoProd(this->beta, y0, pB, eB);
        SpmvTwoSum(hiPtr[j], pB, s, e);
        float lo = loPtr[j] + eB + e;
        outPtr[j] = s + lo;
        if constexpr (!std::is_same_v<OutT, float>) {
            // 16bit 输出：s/lo 回写供中点重判（hiPtr = s，loPtr = lo）
            hiPtr[j] = s;
            loPtr[j] = lo;
        }
    }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::RneMidpointFix16(
    uint32_t chunkSize, const __ubuf__ float *hiPtr, const __ubuf__ float *loPtr,
    const __ubuf__ float *stgPtr) {
    AscendC::LocalTensor<float> tmpT = bufColIdx.Get<float>();
    AscendC::LocalTensor<float> tmpU = bufValsC.Get<float>();
    AscendC::LocalTensor<float> tmpV = bufXRow.Get<float>();
    AscendC::LocalTensor<OutT> yChunk = bufYRaw.Get<OutT>();
    AscendC::LocalTensor<float> stgHi = bufYAcc.Get<float>();
    // 标量写 stgPtr(tmpV)→矢量 Cast 读（RAW）：无细粒度屏障可用，保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Cast<OutT, float>(yChunk, tmpV, AscendC::RoundMode::CAST_RINT, chunkSize);
    // 矢量写 yChunk → 矢量 Cast 读（同管程序序，屏障作防御）
    AscendC::PipeBarrier<PIPE_V>();
    // bisheng 不支持标量 bf16 转换：16bit→f32 一律矢量 Cast staging
    //（tmpT = cast 后结果 yf，tmpU = 邻居值 nf，bufYAcc 复用为邻居位型暂存）
    AscendC::Cast<float, OutT>(tmpT, yChunk, AscendC::RoundMode::CAST_NONE, chunkSize);
    // 矢量写 tmpT → 标量读 yfPtr/yBits
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float *yfPtr = reinterpret_cast<__ubuf__ float *>(tmpT.GetPhyAddr());
    __ubuf__ uint16_t *yBits = reinterpret_cast<__ubuf__ uint16_t *>(yChunk.GetPhyAddr());
    __ubuf__ uint16_t *nbBits = reinterpret_cast<__ubuf__ uint16_t *>(stgHi.GetPhyAddr());
    AscendC::LocalTensor<OutT> nbT = bufYAcc.Get<OutT>();
    SpmvRnePickNeighbors<OutT>(chunkSize, loPtr, stgPtr, yfPtr, yBits, nbBits);
    // 标量写 nbBits→矢量 Cast 读（RAW）：无细粒度屏障可用，保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Cast<float, OutT>(tmpU, nbT, AscendC::RoundMode::CAST_NONE, chunkSize);
    // 矢量写 tmpU → 标量读 nfPtr
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float *nfPtr = reinterpret_cast<__ubuf__ float *>(tmpU.GetPhyAddr());
    SpmvRneMidpointRecheck<OutT>(chunkSize, hiPtr, loPtr, stgPtr, yfPtr, nfPtr, yBits, nbBits);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Process() {
    if constexpr (!DS_WS) {
        ScaleBeta();
        AscendC::SyncAll();
    }
    // DS_WS：不做 beta 预缩放（beta 在 finalize 里 double-single 结合），
    // 不做原子加（每核写自己专属的 ws 平面，跨核顺序无关）

    AscendC::LocalTensor<CompT> yAcc = bufYAcc.Get<CompT>();
    const uint32_t winCountTotal =
        (this->totalColsNum + YWIN_ELEMS - 1) / YWIN_ELEMS;

    for (uint32_t w = 0; w < winCountTotal; w++) {
        const uint32_t winStart = w * YWIN_ELEMS;
        uint32_t winElems = this->totalColsNum - winStart;
        if (winElems > YWIN_ELEMS)
            winElems = YWIN_ELEMS;

        AscendC::Duplicate(yAcc, static_cast<CompT>(0), winElems);
        if constexpr (DS_WS) {
            AscendC::LocalTensor<float> yAccLo = bufYAccLo.Get<float>();
            AscendC::Duplicate(yAccLo, 0.0f, winElems);
        }
        // 矢量清零 yAcc/yAccLo → 标量行段累加读改写
        AscendC::PipeBarrier<PIPE_V>();
        if (this->blockRowNum > 0) {
            if constexpr (DS_WS) {
                AccumulateWindowDs(winStart, winStart + winElems);
            } else {
                AccumulateWindow(winStart, winStart + winElems);
            }
        }

        // 标量写 yAcc(/yAccLo)→MTE3/原子写出：无细粒度屏障可用，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (DS_WS) {
            // 本核专属平面普通写出 (acc, accLo)；行数少于核数时空载核也写零平面，
            // 保证 finalize 读到的每组数据都是本轮写入的
            if (AscendC::GetBlockIdx() < this->dsGroups) {
                AscendC::LocalTensor<float> yAccLo = bufYAccLo.Get<float>();
                uint64_t hiBase = static_cast<uint64_t>(AscendC::GetBlockIdx()) * this->dsStride + winStart;
                uint64_t loBase = static_cast<uint64_t>(this->dsGroups + AscendC::GetBlockIdx()) * this->dsStride + winStart;
                AscendC::DataCopyExtParams cp{1, (uint32_t)(winElems * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(wsGm[hiBase], yAcc, cp);
                AscendC::DataCopyPad(wsGm[loBase], yAccLo, cp);
            }
        }
        else {
            AscendC::SetAtomicAdd<OutT>();
            AscendC::DataCopyExtParams cp{1, (uint32_t)(winElems * sizeof(OutT)), 0, 0, 0};
            AscendC::LocalTensor<OutT> yAccOut = bufYAcc.Get<OutT>();
            AscendC::DataCopyPad(yVecGm[winStart], yAccOut, cp);
            AscendC::SetAtomicNone();
        }
        // MTE3/原子写 yAcc→次窗口矢量清零覆写 WAR，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    if constexpr (DS_WS) {
        AscendC::SyncAll();
        FinalizeDs();
    }
}

/**
 * @brief complex64 SpMV（非转置）。
 *
 * complex64 按交错 float2 存储（re, im）。INPUT_COMPLEX = true：values/x 均为复数；
 * INPUT_COMPLEX = false：values/x 为 FP32 实数，输出复数（虚部由 alpha/beta 复数标量产生）。
 * 累加在 FP32 分量上进行，精度按实/虚部分别满足 FP32 标准。
 * x 以 float 视图按窗口预加载到 UB，窗口边界按 2 的倍数对齐，不会拆散复数元素。
 */
template <bool INPUT_COMPLEX>
class SpmvKernelC64 {
public:
    // x 窗口 float 数（complex 时一个元素占 2 个 float）。
    // 复数输入 28672 float（14336 列，28672 列矩阵 2 窗口）；实数输入 36864 float
    //（28672 列单窗口覆盖 P 场景）。窗口边界均为偶数，不拆散复数元素。
    static constexpr uint32_t XWIN_FLOATS = INPUT_COMPLEX ? 28672 : 36864;

    __aicore__ inline SpmvKernelC64(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        float alphaRe, float alphaIm, float betaRe, float betaIm,
        uint32_t indexBase, uint32_t readY);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BeginGroup(uint32_t groupStart, uint32_t groupCount,
                                      int32_t &groupNnz0, int32_t &groupNnz1);
    // 均匀行长检测：全组行长均为 2 的幂 L（8<=L<=TILE 且对齐组起点）时返回 L，否则 0
    __aicore__ inline int32_t DetectUniformRowLen(uint32_t groupCount, int32_t groupNnz0);
    __aicore__ inline void LoadXWindow(uint32_t winStart, uint32_t winFloats);
    // 软件流水预取一拍 colIdx/valsF（colIdx/valsF 在矢量阶段后不再被读）
    __aicore__ inline void PrefetchNzTile(int32_t nextT0, int32_t groupNnz1);
    // 矢量聚集本拍积：gather 偏移以字节为单位；多窗口时钳位 + Compare 掩码 + Select 置零
    __aicore__ inline void GatherProdTile(uint32_t winStart, uint32_t winFloats,
                                          uint32_t winCountTotal, int32_t valid);
    // indexBase 与窗口起点合并移位 + 多窗口钳位（同 V 流水按发射序执行，无需 barrier）
    __aicore__ inline void ClampColIdxForMultiWin(const AscendC::LocalTensor<int32_t> &colIdx,
                                                  const AscendC::LocalTensor<int32_t> &prodI32,
                                                  const AscendC::LocalTensor<uint8_t> &inWin,
                                                  uint32_t winCElems, int32_t valid,
                                                  int32_t colShift, bool multiWin);
    // 复数积 (vr + i*vi)(xr + i*xi) → (prodR, prodI)；vi 用完后复用为暂存
    __aicore__ inline void ComplexMulProdTile(const AscendC::LocalTensor<float> &prodR,
                                              const AscendC::LocalTensor<float> &prodI,
                                              const AscendC::LocalTensor<float> &vr,
                                              const AscendC::LocalTensor<float> &vi,
                                              const AscendC::LocalTensor<float> &xr,
                                              const AscendC::LocalTensor<float> &xi,
                                              int32_t valid);
    // 均匀行长求和：log2(L/8) 次整块 Add 对数折叠 + 块首 8 元素标量收尾
    __aicore__ inline void SumUniformTile(int32_t valid, int32_t uniformL, int32_t groupNnz0,
                                          int32_t t0);
    // 非均匀行段求和：多累加器标量展开（打断串行 FP 加法依赖链）
    __aicore__ inline void SumGenericTile(int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount);
    __aicore__ inline void FinalizeGroup(uint32_t groupStart, uint32_t groupCount);
    __aicore__ inline void LoadNzTile(int32_t t0, int32_t valid);

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRowPtr, bufAccR, bufAccI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufColIdx, bufValsF;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXWin, bufYTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufProdR, bufProdI;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXr, bufXi, bufVr, bufVi;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufWk, bufIota;
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<float> csrValFGm; // 复数视图：2*nnz 个 float；实数输入时为 nnz 个 float
    AscendC::GlobalTensor<float> xVecFGm;   // 复数视图：2*cols；实数输入时为 cols
    AscendC::GlobalTensor<float> yVecFGm;   // 复数视图：2*rows

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t startValIdx;
    uint32_t endValIdx;
    float alphaRe, alphaIm, betaRe, betaIm;
    uint32_t indexBase;
    uint32_t readY;
};

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    float alphaRe, float alphaIm, float betaRe, float betaIm,
    uint32_t indexBase, uint32_t readY) {
    SpmvInitCommon(totalRowsNum, totalColsNum, indexBase, readY,
                   this->totalRowsNum, this->totalColsNum, this->indexBase, this->readY);
    SpmvInitC64Scalars(alphaRe, alphaIm, betaRe, betaIm,
                       this->alphaRe, this->alphaIm, this->betaRe, this->betaIm);

    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0) {
        this->blockRowNum = 0;
        return;
    }
    // 均分行：前 tempRowNum 个核各多拿一行
    SpmvSplitBlockRange(totalRowsNum, blockNum, this->blockRowNum, this->startRow);

    xVecFGm.SetGlobalBuffer((__gm__ float *)xVec,
                            this->totalColsNum * (INPUT_COMPLEX ? 2 : 1));
    yVecFGm.SetGlobalBuffer((__gm__ float *)yVec + this->startRow * 2, this->blockRowNum * 2);

    SpmvBindRowRange(csrRowPtrGm, csrRowPtr, this->startRow, this->blockRowNum, this->indexBase,
                     this->startValIdx, this->endValIdx);
    if (this->blockRowNum > 0) {
        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + this->startValIdx,
                                    this->endValIdx - this->startValIdx);
        csrValFGm.SetGlobalBuffer((__gm__ float *)csrVal + this->startValIdx * (INPUT_COMPLEX ? 2 : 1),
                                  (this->endValIdx - this->startValIdx) * (INPUT_COMPLEX ? 2 : 1));
    }

    pipe.InitBuffer(bufRowPtr, (SPMV_ROW_GROUP + 1) * sizeof(int32_t));
    pipe.InitBuffer(bufAccR, SPMV_ROW_GROUP * sizeof(float));
    pipe.InitBuffer(bufAccI, SPMV_ROW_GROUP * sizeof(float));
    pipe.InitBuffer(bufColIdx, SPMV_TILE_ELEMS_C64 * sizeof(int32_t));
    pipe.InitBuffer(bufValsF, SPMV_TILE_ELEMS_C64 * (INPUT_COMPLEX ? 2 : 1) * sizeof(float));
    pipe.InitBuffer(bufXWin, XWIN_FLOATS * sizeof(float));
    pipe.InitBuffer(bufYTile, 2 * SPMV_ROW_GROUP * sizeof(float));
    // 矢量 gather/复数乘的暂存；实数输入时 xi/vi/prodI 恒零不分配
    pipe.InitBuffer(bufProdR, SPMV_TILE_ELEMS_C64 * sizeof(float));
    pipe.InitBuffer(bufProdI, INPUT_COMPLEX ? SPMV_TILE_ELEMS_C64 * sizeof(float) : 32);
    pipe.InitBuffer(bufXr, SPMV_TILE_ELEMS_C64 * sizeof(float));
    pipe.InitBuffer(bufXi, INPUT_COMPLEX ? SPMV_TILE_ELEMS_C64 * sizeof(float) : 32);
    pipe.InitBuffer(bufVr, INPUT_COMPLEX ? SPMV_TILE_ELEMS_C64 * sizeof(float) : 32);
    pipe.InitBuffer(bufVi, INPUT_COMPLEX ? SPMV_TILE_ELEMS_C64 * sizeof(float) : 32);
    // 前 1KB 与多窗口掩码分时复用
    pipe.InitBuffer(bufWk, 1024);
    // vals 解交错用的字节偏移 iota（8j 与 8j+4），一次构建全程复用
    pipe.InitBuffer(bufIota, INPUT_COMPLEX ? 2 * SPMV_TILE_ELEMS_C64 * sizeof(int32_t) : 32);
}

// 加载一拍 colIdx/vals（供软件流水预取；调用方负责同步）
template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::LoadNzTile(int32_t t0, int32_t valid) {
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<float> valsF = bufValsF.Get<float>();
    AscendC::DataCopyExtParams cpI{1, (uint32_t)(valid * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
    AscendC::DataCopyPad(colIdx, csrColIndGm[t0 - static_cast<int32_t>(this->startValIdx)], cpI,
                         padI);
    const uint32_t valFloats = valid * (INPUT_COMPLEX ? 2 : 1);
    AscendC::DataCopyExtParams cpV{1, (uint32_t)(valFloats * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padV{false, 0, 0, 0};
    AscendC::DataCopyPad(
        valsF, csrValFGm[(t0 - static_cast<int32_t>(this->startValIdx)) * (INPUT_COMPLEX ? 2 : 1)],
        cpV, padV);
}

// 行组初始化：rowPtr 批量入 UB + accR/accI 清零 + 本组 nnz 区间
template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::BeginGroup(
    uint32_t groupStart, uint32_t groupCount, int32_t &groupNnz0, int32_t &groupNnz1) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    SpmvLoadGroupRowPtr(csrRowPtrGm, rowPtr, groupStart, groupCount);
    AscendC::LocalTensor<float> accR = bufAccR.Get<float>();
    AscendC::LocalTensor<float> accI = bufAccI.Get<float>();
    AscendC::Duplicate(accR, 0.0f, groupCount);
    AscendC::Duplicate(accI, 0.0f, groupCount);
    // 矢量清零 accR/accI → 后续标量读改写（行段求和）；中间隔着 LoadXWindow/拍头的全屏障，PIPE_V 已足够
    AscendC::PipeBarrier<PIPE_V>();
    SpmvGroupNnzRange(rowPtr, groupCount, this->indexBase, groupNnz0, groupNnz1);
}

template <bool INPUT_COMPLEX>
__aicore__ inline int32_t SpmvKernelC64<INPUT_COMPLEX>::DetectUniformRowLen(
    uint32_t groupCount, int32_t groupNnz0) {
    // 均匀行长检测：全组行长均为 2 的幂 L（8<=L<=TILE 且对齐组起点）时，行段求和可用
    // 对数折叠（log2(L/8) 次整块 Add）替代逐元素标量累加——实测 UB 标量 load+add
    // 约 10-30cyc/元素，是 c64 N 路径主瓶颈
    int32_t uniformL = 0;
    if (groupCount > 0) {
        AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
        __ubuf__ int32_t *rpw =
            reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
        const int32_t l0 = rpw[1] - rpw[0];
        if (l0 >= 8 && l0 <= static_cast<int32_t>(SPMV_TILE_ELEMS_C64) &&
            (l0 & (l0 - 1)) == 0 && (groupNnz0 % l0) == 0) {
            uniformL = l0;
            for (uint32_t r = 1; r < groupCount; r++) {
                if (rpw[r + 1] - rpw[r] != l0) {
                    uniformL = 0;
                    break;
                }
            }
        }
    }
    return uniformL;
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::LoadXWindow(
    uint32_t winStart, uint32_t winFloats) {
    AscendC::LocalTensor<float> xWin = bufXWin.Get<float>();
    // WAR：等上一窗口矢量 Gather 读完 xWin，再由 MTE2 覆写
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopyExtParams cpX{1, (uint32_t)(winFloats * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padX{false, 0, 0, 0};
    AscendC::DataCopyPad(xWin, xVecFGm[winStart], cpX, padX);
    // MTE2→矢量 Gather：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::PrefetchNzTile(
    int32_t nextT0, int32_t groupNnz1) {
    if (nextT0 < groupNnz1) {
        int32_t validN = groupNnz1 - nextT0;
        if (validN > static_cast<int32_t>(SPMV_TILE_ELEMS_C64))
            validN = SPMV_TILE_ELEMS_C64;
        LoadNzTile(nextT0, validN);
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::ClampColIdxForMultiWin(
    const AscendC::LocalTensor<int32_t> &colIdx,
    const AscendC::LocalTensor<int32_t> &prodI32,
    const AscendC::LocalTensor<uint8_t> &inWin,
    uint32_t winCElems, int32_t valid, int32_t colShift, bool multiWin) {
    if (colShift != 0) {
        AscendC::Adds(colIdx, colIdx, -colShift, valid);
    }
    if (multiWin) {
        AscendC::Adds(prodI32, colIdx, 0, valid); // 备份钳位前偏移
        AscendC::Maxs(colIdx, colIdx, 0, valid);
        AscendC::Mins(colIdx, colIdx, static_cast<int32_t>(winCElems) - 1, valid);
        AscendC::Compare(inWin, prodI32, colIdx, AscendC::CMPMODE::EQ,
                         static_cast<uint32_t>(valid));
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::ComplexMulProdTile(
    const AscendC::LocalTensor<float> &prodR, const AscendC::LocalTensor<float> &prodI,
    const AscendC::LocalTensor<float> &vr, const AscendC::LocalTensor<float> &vi,
    const AscendC::LocalTensor<float> &xr, const AscendC::LocalTensor<float> &xi,
    int32_t valid) {
    AscendC::Mul(prodR, vr, xr, valid);
    AscendC::Mul(prodI, vi, xi, valid);
    AscendC::Sub(prodR, prodR, prodI, valid);
    AscendC::Mul(prodI, vr, xi, valid);
    AscendC::Mul(vi, vi, xr, valid); // vi 已用完，复用为暂存
    AscendC::Add(prodI, prodI, vi, valid);
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::GatherProdTile(
    uint32_t winStart, uint32_t winFloats, uint32_t winCountTotal, int32_t valid) {
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<float> valsF = bufValsF.Get<float>();
    AscendC::LocalTensor<float> xWin = bufXWin.Get<float>();
    const uint32_t winCElems = INPUT_COMPLEX ? (winFloats >> 1) : winFloats;
    const uint32_t winCStart = INPUT_COMPLEX ? (winStart >> 1) : winStart;
    AscendC::LocalTensor<float> prodR = bufProdR.Get<float>();
    AscendC::LocalTensor<float> xr = bufXr.Get<float>();
    AscendC::LocalTensor<int32_t> prodI32 = prodR.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<uint8_t> inWin = bufWk.Get<uint8_t>();
    const bool multiWin = winCountTotal > 1;
    // indexBase 与窗口起点合并为一次移位（同 V 流水按发射序执行，无需 barrier）
    const int32_t colShift = static_cast<int32_t>(this->indexBase) +
        (multiWin ? static_cast<int32_t>(winCStart) : 0);
    if constexpr (INPUT_COMPLEX) {
        // valsF 解交错：vr/vi 分别用预构建的字节偏移 8j / 8j+4 gather
        AscendC::LocalTensor<float> vr = bufVr.Get<float>();
        AscendC::LocalTensor<float> vi = bufVi.Get<float>();
        AscendC::LocalTensor<uint32_t> iotaU = bufIota.Get<uint32_t>();
        AscendC::Gather(vr, valsF, iotaU, 0, static_cast<uint32_t>(valid));
        AscendC::Gather(vi, valsF, iotaU[SPMV_TILE_ELEMS_C64], 0,
                        static_cast<uint32_t>(valid));

        AscendC::LocalTensor<float> xi = bufXi.Get<float>();
        AscendC::LocalTensor<float> prodI = bufProdI.Get<float>();
        ClampColIdxForMultiWin(colIdx, prodI32, inWin, winCElems, valid, colShift, multiWin);
        AscendC::Muls(colIdx, colIdx, 8, valid);
        AscendC::LocalTensor<uint32_t> offsX = colIdx.ReinterpretCast<uint32_t>();
        AscendC::Gather(xr, xWin, offsX, 0, static_cast<uint32_t>(valid));
        // xi 与 xr 同偏移表，srcBaseOffset 加 4 字节取虚部，省一次 Adds
        AscendC::Gather(xi, xWin, offsX, 4, static_cast<uint32_t>(valid));
        if (multiWin) {
            AscendC::Select(xr, inWin, xr, 0.0f,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            static_cast<uint32_t>(valid));
            AscendC::Select(xi, inWin, xi, 0.0f,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            static_cast<uint32_t>(valid));
        }
        // 复数积：(vr + i*vi)(xr + i*xi)
        ComplexMulProdTile(prodR, prodI, vr, vi, xr, xi, valid);
    }
    else {
        // 实数输入：prodR = x[c] * v，accI 恒 0
        ClampColIdxForMultiWin(colIdx, prodI32, inWin, winCElems, valid, colShift, multiWin);
        AscendC::Muls(colIdx, colIdx, 4, valid);
        AscendC::LocalTensor<uint32_t> offsX = colIdx.ReinterpretCast<uint32_t>();
        AscendC::Gather(xr, xWin, offsX, 0, static_cast<uint32_t>(valid));
        if (multiWin) {
            AscendC::Select(xr, inWin, xr, 0.0f,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            static_cast<uint32_t>(valid));
        }
        AscendC::Mul(prodR, xr, valsF, valid);
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::SumUniformTile(
    int32_t valid, int32_t uniformL, int32_t groupNnz0, int32_t t0) {
    AscendC::LocalTensor<float> prodR = bufProdR.Get<float>();
    AscendC::LocalTensor<float> prodI = bufProdI.Get<float>();
    __ubuf__ float *prPtr = reinterpret_cast<__ubuf__ float *>(prodR.GetPhyAddr());
    __ubuf__ float *piPtr = reinterpret_cast<__ubuf__ float *>(prodI.GetPhyAddr());
    __ubuf__ float *arPtr = reinterpret_cast<__ubuf__ float *>(bufAccR.Get<float>().GetPhyAddr());
    __ubuf__ float *aiPtr = reinterpret_cast<__ubuf__ float *>(bufAccI.Get<float>().GetPhyAddr());
    // 对数折叠：第 s 次折叠后 [0,s) 为跨距 s 的有效部分和，死区数据不再被读故无需掩码
    for (int32_t s = uniformL >> 1; s >= 8; s >>= 1) {
        AscendC::Add(prodR, prodR, prodR[s], valid - s);
        if constexpr (INPUT_COMPLEX) {
            AscendC::Add(prodI, prodI, prodI[s], valid - s);
        }
    }
    // 矢量折叠写 prodR/prodI → 标量读 prPtr/piPtr
    AscendC::PipeBarrier<PIPE_V>();
    // 均匀路径无需读 rowPtr：行号与块内偏移直接由 L 算出
    const uint32_t rowBase =
        static_cast<uint32_t>(t0 - groupNnz0) / static_cast<uint32_t>(uniformL);
    uint32_t r = rowBase;
    for (int32_t off = 0; off < valid; off += uniformL, r++) {
        arPtr[r] = arPtr[r] + ((prPtr[off] + prPtr[off + 1]) +
                               (prPtr[off + 2] + prPtr[off + 3]) +
                               (prPtr[off + 4] + prPtr[off + 5]) +
                               (prPtr[off + 6] + prPtr[off + 7]));
        if constexpr (INPUT_COMPLEX) {
            aiPtr[r] = aiPtr[r] + ((piPtr[off] + piPtr[off + 1]) +
                                   (piPtr[off + 2] + piPtr[off + 3]) +
                                   (piPtr[off + 4] + piPtr[off + 5]) +
                                   (piPtr[off + 6] + piPtr[off + 7]));
        }
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::SumGenericTile(
    int32_t t0, int32_t t1, uint32_t row, uint32_t groupCount) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<float> prodR = bufProdR.Get<float>();
    AscendC::LocalTensor<float> prodI = bufProdI.Get<float>();
    __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
    __ubuf__ float *prPtr = reinterpret_cast<__ubuf__ float *>(prodR.GetPhyAddr());
    __ubuf__ float *piPtr = reinterpret_cast<__ubuf__ float *>(prodI.GetPhyAddr());
    __ubuf__ float *arPtr = reinterpret_cast<__ubuf__ float *>(bufAccR.Get<float>().GetPhyAddr());
    __ubuf__ float *aiPtr = reinterpret_cast<__ubuf__ float *>(bufAccI.Get<float>().GetPhyAddr());
    // 多累加器展开：打断串行 FP 加法依赖链（UB 标量 load->add 延迟
    // 约 10cyc，单累加器逐元素求和是 c64 N 路径主瓶颈）
    SpmvRowSegCursor cur;
    cur.Init(rpPtr, this->indexBase, row, groupCount, t0, t1);
    uint32_t r;
    while (cur.Next(r)) {
        int32_t len = cur.hi - cur.lo;
        const int32_t off = cur.lo - t0;
        int32_t j = 0;
        float sr0 = 0.0f, sr1 = 0.0f, sr2 = 0.0f, sr3 = 0.0f;
        if constexpr (INPUT_COMPLEX) {
            float si0 = 0.0f, si1 = 0.0f, si2 = 0.0f, si3 = 0.0f;
            for (; j + 4 <= len; j += 4) {
                sr0 += prPtr[off + j];     si0 += piPtr[off + j];
                sr1 += prPtr[off + j + 1]; si1 += piPtr[off + j + 1];
                sr2 += prPtr[off + j + 2]; si2 += piPtr[off + j + 2];
                sr3 += prPtr[off + j + 3]; si3 += piPtr[off + j + 3];
            }
            for (; j < len; j++) {
                sr0 += prPtr[off + j];
                si0 += piPtr[off + j];
            }
            aiPtr[r] = aiPtr[r] + (si0 + si1) + (si2 + si3);
        }
        else {
            for (; j + 4 <= len; j += 4) {
                sr0 += prPtr[off + j];
                sr1 += prPtr[off + j + 1];
                sr2 += prPtr[off + j + 2];
                sr3 += prPtr[off + j + 3];
            }
            for (; j < len; j++) {
                sr0 += prPtr[off + j];
            }
        }
        arPtr[r] = arPtr[r] + (sr0 + sr1) + (sr2 + sr3);
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::FinalizeGroup(uint32_t groupStart, uint32_t groupCount) {
    // accR/accI 的前序写均为标量（随程序序完成），此处无需屏障
    AscendC::LocalTensor<float> accR = bufAccR.Get<float>();
    AscendC::LocalTensor<float> accI = bufAccI.Get<float>();
    AscendC::LocalTensor<float> yTile = bufYTile.Get<float>();

    if (this->readY != 0) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(2 * groupCount * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(yTile, yVecFGm[groupStart * 2], cp, pad);
        // MTE2→标量读 yTile
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    AscendC::LocalTensor<float> outBuf = bufXWin.Get<float>(); // x 窗口已用完，复用其内存
    for (uint32_t r = 0; r < groupCount; r++) {
        float ar = accR.GetValue(r);
        float ai = accI.GetValue(r);
        float outR = this->alphaRe * ar - this->alphaIm * ai;
        float outI = this->alphaRe * ai + this->alphaIm * ar;
        if (this->readY != 0) {
            float yr = yTile.GetValue(2 * r);
            float yi = yTile.GetValue(2 * r + 1);
            outR += this->betaRe * yr - this->betaIm * yi;
            outI += this->betaRe * yi + this->betaIm * yr;
        }
        outBuf.SetValue(2 * r, outR);
        outBuf.SetValue(2 * r + 1, outI);
    }
    // 标量写 outBuf→MTE3 写出：无细粒度屏障可用，保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp{1, (uint32_t)(2 * groupCount * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(yVecFGm[groupStart * 2], outBuf, cp);
    // MTE3 读 outBuf→次组 xWin 覆写 WAR（outBuf 复用 bufXWin），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64<INPUT_COMPLEX>::Process() {
    if (this->blockRowNum == 0)
        return;

    // 窗口 float 总数（复数时 2*cols，窗口按偶数边界对齐，不拆散复数）
    const uint32_t xFloats = this->totalColsNum * (INPUT_COMPLEX ? 2 : 1);
    const uint32_t winCountTotal = (xFloats + XWIN_FLOATS - 1) / XWIN_FLOATS;

    if constexpr (INPUT_COMPLEX) {
        // vals 解交错字节偏移：iota8 = 8j 与 8j+4，一次构建全程复用
        AscendC::LocalTensor<int32_t> io = bufIota.Get<int32_t>();
        AscendC::CreateVecIndex(io, 0, SPMV_TILE_ELEMS_C64);
        AscendC::Muls(io, io, 8, SPMV_TILE_ELEMS_C64);
        AscendC::Adds(io[SPMV_TILE_ELEMS_C64], io, 4, SPMV_TILE_ELEMS_C64);
        // 矢量写 iota → 矢量 Gather 以其为偏移
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        uint32_t groupCount = this->blockRowNum - groupStart;
        if (groupCount > SPMV_ROW_GROUP)
            groupCount = SPMV_ROW_GROUP;

        int32_t groupNnz0, groupNnz1;
        BeginGroup(groupStart, groupCount, groupNnz0, groupNnz1);
        const int32_t uniformL = DetectUniformRowLen(groupCount, groupNnz0);

        for (uint32_t w = 0; w < winCountTotal; w++) {
            const uint32_t winStart = w * XWIN_FLOATS;
            uint32_t winFloats = xFloats - winStart;
            if (winFloats > XWIN_FLOATS)
                winFloats = XWIN_FLOATS;
            const uint32_t winEnd = winStart + winFloats;

            LoadXWindow(winStart, winFloats);

            uint32_t row = 0;
            // 软件流水：先预取首拍；每拍矢量算完即发射下一拍 DMA（colIdx/valsF 在
            // 矢量阶段后不再被读），让 DMA 与行段求和重叠，隐藏 GM 访存延迟
            int32_t t0 = groupNnz0;
            PrefetchNzTile(t0, groupNnz1);
            for (; t0 < groupNnz1; t0 += SPMV_TILE_ELEMS_C64) {
                int32_t valid = groupNnz1 - t0;
                if (valid > static_cast<int32_t>(SPMV_TILE_ELEMS_C64))
                    valid = SPMV_TILE_ELEMS_C64;
                const int32_t t1 = t0 + valid;

                // 等本拍 DMA 完成：MTE2→矢量 Gather，实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
                GatherProdTile(winStart, winFloats, winCountTotal, valid);
                // Gather 的写在 arch22 上不被 PipeBarrier<PIPE_V> 覆盖（实测确定性失败），保留全屏障
                AscendC::PipeBarrier<PIPE_ALL>();
                // 预取下一拍（colIdx/valsF 在矢量阶段后不再被读，DMA 与下方求和重叠）
                PrefetchNzTile(t1, groupNnz1);

                // 按行段求和进 accR/accI：均匀行长（组级检测 uniformL）时用对数折叠；
                // 否则回退多累加器标量求和（行段游标跳过整段在 tile 前的行）
                if (uniformL > 0) {
                    SumUniformTile(valid, uniformL, groupNnz0, t0);
                }
                else {
                    SumGenericTile(t0, t1, row, groupCount);
                }
            }
        }
        FinalizeGroup(groupStart, groupCount);
    }
}

/**
 * @brief complex64 SpMV（转置）。beta 缩放按列均分（UB 内标量缩放 + 整块 DMA）；
 * 非零元贡献先聚合到 UB 的 y 累加窗口（交错 float2），窗口结束后以块级
 * SetAtomicAdd + DataCopyPad 合并到 y，避免逐元素原子 DMA。
 */
template <bool INPUT_COMPLEX>
class SpmvKernelC64Trans {
public:
    // y 累加窗口 float 数（一个复数列占 2 个 float）
    static constexpr uint32_t YWIN_FLOATS = 16384;

    __aicore__ inline SpmvKernelC64Trans(){};

    __aicore__ inline void Init(
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
        GM_ADDR ws,
        uint32_t totalRowsNum, uint32_t totalColsNum,
        float alphaRe, float alphaIm, float betaRe, float betaIm,
        uint32_t indexBase, uint32_t readY);
    __aicore__ inline void Process();
    // 单行段贡献（复数乘 + alpha 复数标量乘）累加进 yAcc 交错窗口
    //（public：供文件级共用模板 SpmvAccumulateTileSegs(Ds) 回调）
    __aicore__ inline void AccumulateSeg(uint32_t r, int32_t lo, int32_t hi,
                                         const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr,
                                         const __ubuf__ float *xRowPtr, __ubuf__ float *yAccPtr,
                                         int32_t t0, uint32_t winStart, uint32_t winEnd);
    // 复数 Dot2 单行段：贡献 alpha*(v*x) 的每个实数乘法用 Dekker 精确积，
    // 加减用 Knuth TwoSum，(acc, accLo) 双字窗口按交错 float2 分量独立累加。
    // GUARDED=false 为无守卫快路径（调用方谓词已保证本行全部乘积安全）
    template <bool GUARDED>
    __aicore__ inline void AccumulateSegDs(uint32_t r, int32_t lo, int32_t hi,
                                           const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr,
                                           const __ubuf__ float *xRowPtr, __ubuf__ float *accPtr,
                                           __ubuf__ float *accLoPtr, int32_t t0,
                                           uint32_t winStart, uint32_t winEnd);
    // 行级安全谓词（守卫外提）：行段内（限窗口）max|val 分量| 与本行 max|x 分量|
    // 的上界乘积覆盖复乘 4 个实数积与 alpha 结合 4 个实数积的全部溢出风险
    __aicore__ inline bool RowDsSafe(uint32_t r, int32_t lo, int32_t hi,
                                     const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr,
                                     const __ubuf__ float *xRowPtr, int32_t t0,
                                     uint32_t winStart, uint32_t winEnd, float absAlphaMax);

private:
    __aicore__ inline void ScaleBeta();
    // 本组行的 x 批量入 UB（复数输入 2*count 个 float）
    __aicore__ inline void LoadXGroup(uint32_t globalRow, uint32_t count);
    // 加载一拍 colIdx / vals tile（实数输入 vals 为 valid 个 float，复数为 2*valid）
    __aicore__ inline void LoadNzTile(int32_t t0, int32_t valid);
    // 行组头：rowPtr 批量入 UB + 本组 x 入 UB + 本组 nnz 区间
    __aicore__ inline SpmvGroupHead BeginGroup(uint32_t groupStart);
    __aicore__ inline void AccumulateWindow(uint32_t winStart, uint32_t winEnd);
    __aicore__ inline void AccumulateWindowDs(uint32_t winStart, uint32_t winEnd);
    __aicore__ inline void FinalizeDs();
    // 矢量 TwoSum 合并第 g 组 (hi, lo) 平面（分量独立）
    __aicore__ inline void MergeGroupPlanes(uint32_t g, uint32_t chunkFloats, uint32_t col0,
                                            uint32_t accOff, uint32_t tmpVOff);
    // beta*y0 复数 double-single 乘 + acc，单次舍入
    __aicore__ inline void CombineBetaY0(uint32_t chunkCols, const __ubuf__ float *hiPtr,
                                         const __ubuf__ float *loPtr, const __ubuf__ float *y0Ptr,
                                         __ubuf__ float *outPtr);

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRowPtr, bufXRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufColIdx, bufValsF;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYAcc, bufYAccLo;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRedWk; // 守卫外提谓词的归约工作区（1KB）
    AscendC::GlobalTensor<int32_t> csrRowPtrGm;
    AscendC::GlobalTensor<int32_t> csrColIndGm;
    AscendC::GlobalTensor<float> csrValFGm;
    AscendC::GlobalTensor<float> xVecFGm; // 复数视图：2*rows；实数输入时为 rows
    AscendC::GlobalTensor<float> yVecFGm; // 复数视图：2*cols
    AscendC::GlobalTensor<float> wsGm;    // double-single 平面：groups 个 hi + groups 个 lo

    uint32_t totalRowsNum;
    uint32_t totalColsNum;
    uint32_t startRow;
    uint32_t blockRowNum;
    uint32_t startValIdx;
    uint32_t endValIdx;
    float alphaRe, alphaIm, betaRe, betaIm;
    uint32_t indexBase;
    uint32_t readY;
    uint32_t yBlockStart;
    uint32_t yBlockCount;
    uint32_t dsGroups;   // 参与累加的核组数 = min(blockNum, max(rows,1))
    uint32_t dsStride;   // 每核每平面 float 数 = roundUp(2*cols, 8)
};

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
    GM_ADDR ws,
    uint32_t totalRowsNum, uint32_t totalColsNum,
    float alphaRe, float alphaIm, float betaRe, float betaIm,
    uint32_t indexBase, uint32_t readY) {
    SpmvInitCommon(totalRowsNum, totalColsNum, indexBase, readY,
                   this->totalRowsNum, this->totalColsNum, this->indexBase, this->readY);
    SpmvInitC64Scalars(alphaRe, alphaIm, betaRe, betaIm,
                       this->alphaRe, this->alphaIm, this->betaRe, this->betaIm);

    uint32_t blockNum = AscendC::GetBlockNum();
    {
        uint32_t rr = totalRowsNum > 0 ? totalRowsNum : 1;
        this->dsGroups = blockNum > 0 && blockNum < rr ? blockNum : rr;
        // 每核每平面 float 数（复数 2*cols）按 32B（8 float）对齐
        this->dsStride = (2 * totalColsNum + 7) / 8 * 8;
    }
    if (blockNum > 0) {
        // 均分行：前 tempRowNum 个核各多拿一行
        SpmvSplitBlockRange(totalRowsNum, blockNum, this->blockRowNum, this->startRow);
        // y 有 2*cols 个 float（交错复数），按列均分缩放
        SpmvSplitBlockRange(totalColsNum, blockNum, this->yBlockCount, this->yBlockStart);
    }
    else {
        this->blockRowNum = 0;
        this->yBlockCount = 0;
    }

    xVecFGm.SetGlobalBuffer((__gm__ float *)xVec,
                            this->totalRowsNum * (INPUT_COMPLEX ? 2 : 1));
    yVecFGm.SetGlobalBuffer((__gm__ float *)yVec, this->totalColsNum * 2);
    wsGm.SetGlobalBuffer((__gm__ float *)ws, 2 * this->dsGroups * this->dsStride);

    SpmvBindRowRange(csrRowPtrGm, csrRowPtr, this->startRow, this->blockRowNum, this->indexBase,
                     this->startValIdx, this->endValIdx);
    if (this->blockRowNum > 0) {
        csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + this->startValIdx,
                                    this->endValIdx - this->startValIdx);
        csrValFGm.SetGlobalBuffer((__gm__ float *)csrVal + this->startValIdx * (INPUT_COMPLEX ? 2 : 1),
                                  (this->endValIdx - this->startValIdx) * (INPUT_COMPLEX ? 2 : 1));
    }

    pipe.InitBuffer(bufRowPtr, (SPMV_ROW_GROUP + 1) * sizeof(int32_t));
    pipe.InitBuffer(bufXRow, SPMV_ROW_GROUP * (INPUT_COMPLEX ? 2 : 1) * sizeof(float));
    pipe.InitBuffer(bufColIdx, SPMV_TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufValsF, SPMV_TILE_ELEMS * (INPUT_COMPLEX ? 2 : 1) * sizeof(float));
    pipe.InitBuffer(bufYAcc, YWIN_FLOATS * sizeof(float));
    pipe.InitBuffer(bufYAccLo, YWIN_FLOATS * sizeof(float));
    // 守卫外提谓词的归约工作区（本类无其它可借死区；总 UB 仍低于预算上界）
    pipe.InitBuffer(bufRedWk, 1024);
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::ScaleBeta() {
    if (this->yBlockCount == 0)
        return;
    bool betaOne = (this->betaRe == 1.0f && this->betaIm == 0.0f);
    if (this->readY != 0 && betaOne)
        return;

    // 每列 2 个 float（re, im），按 chunk 批量 DMA 后在 UB 内标量缩放，再整块写回
    AscendC::LocalTensor<float> yChunk = bufYAcc.Get<float>();
    for (uint32_t colOff = 0; colOff < this->yBlockCount; colOff += SPMV_TILE_ELEMS_C64) {
        uint32_t chunkCols = SPMV_TILE_ELEMS_C64;
        if (colOff + chunkCols > this->yBlockCount)
            chunkCols = this->yBlockCount - colOff;
        const uint32_t chunkFloats = 2 * chunkCols;
        const uint32_t gmOff = 2 * (this->yBlockStart + colOff);

        if (this->readY == 0) {
            AscendC::Duplicate(yChunk, 0.0f, chunkFloats);
            // （该函数当前无调用点）矢量写→MTE3 写回，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        else {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkFloats * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(yChunk, yVecFGm[gmOff], cp, padParams);
            // （该函数当前无调用点）MTE2→标量读 yChunk，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < chunkCols; j++) {
                float yr = yChunk.GetValue(2 * j);
                float yi = yChunk.GetValue(2 * j + 1);
                yChunk.SetValue(2 * j, this->betaRe * yr - this->betaIm * yi);
                yChunk.SetValue(2 * j + 1, this->betaRe * yi + this->betaIm * yr);
            }
            // 标量写 yChunk→MTE3 写回：无细粒度屏障可用，保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkFloats * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yVecFGm[gmOff], yChunk, cp2);
        // MTE3 读 yChunk→次 chunk 矢量清零覆写 WAR，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::LoadXGroup(
    uint32_t globalRow, uint32_t count) {
    AscendC::LocalTensor<float> xRow = bufXRow.Get<float>();
    AscendC::DataCopyExtParams cpX{1,
        (uint32_t)(count * (INPUT_COMPLEX ? 2 : 1) * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padX{false, 0, 0, 0};
    AscendC::DataCopyPad(xRow, xVecFGm[globalRow * (INPUT_COMPLEX ? 2 : 1)], cpX, padX);
    // MTE2→标量读 xRow
    AscendC::PipeBarrier<PIPE_MTE2>();
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::LoadNzTile(int32_t t0, int32_t valid) {
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<float> valsF = bufValsF.Get<float>();
    SpmvLoadColIdxTile(csrColIndGm, colIdx, t0, this->startValIdx, valid);
    const uint32_t valFloats = valid * (INPUT_COMPLEX ? 2 : 1);
    AscendC::DataCopyExtParams cpV{1, (uint32_t)(valFloats * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padV{false, 0, 0, 0};
    AscendC::DataCopyPad(valsF,
        csrValFGm[(t0 - static_cast<int32_t>(this->startValIdx)) * (INPUT_COMPLEX ? 2 : 1)],
        cpV, padV);
    // MTE2→矢量 Adds（减基）：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    SpmvApplyIndexBase(colIdx, this->indexBase, valid);
}

// 行组头：rowPtr 批量入 UB + 本组 x 入 UB + 本组 nnz 区间
template <bool INPUT_COMPLEX>
__aicore__ inline SpmvGroupHead SpmvKernelC64Trans<INPUT_COMPLEX>::BeginGroup(
    uint32_t groupStart) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    SpmvGroupHead head;
    head.groupCount = this->blockRowNum - groupStart;
    if (head.groupCount > SPMV_ROW_GROUP)
        head.groupCount = SPMV_ROW_GROUP;
    SpmvLoadGroupRowPtr(csrRowPtrGm, rowPtr, groupStart, head.groupCount);
    LoadXGroup(this->startRow + groupStart, head.groupCount);
    SpmvGroupNnzRange(rowPtr, head.groupCount, this->indexBase, head.groupNnz0, head.groupNnz1);
    return head;
}

// 遍历本核全部行，将落在 [winStart, winEnd) 内的贡献累加到 UB y 窗口（交错 float2）
template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::AccumulateWindow(
    uint32_t winStart, uint32_t winEnd) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<float> xRow = bufXRow.Get<float>();
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<float> valsF = bufValsF.Get<float>();
    AscendC::LocalTensor<float> yAcc = bufYAcc.Get<float>();

    uint32_t row = 0;
    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        row = 0;
        const SpmvGroupHead head = BeginGroup(groupStart);
        const uint32_t groupCount = head.groupCount;
        for (int32_t t0 = head.groupNnz0; t0 < head.groupNnz1; t0 += SPMV_TILE_ELEMS) {
            int32_t valid;
            const int32_t t1 = SpmvBeginTile(t0, head.groupNnz1, valid);
            LoadNzTile(t0, valid);

            __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
            __ubuf__ int32_t *colPtr = reinterpret_cast<__ubuf__ int32_t *>(colIdx.GetPhyAddr());
            __ubuf__ float *valPtr = reinterpret_cast<__ubuf__ float *>(valsF.GetPhyAddr());
            __ubuf__ float *xRowPtr = reinterpret_cast<__ubuf__ float *>(xRow.GetPhyAddr());
            __ubuf__ float *yAccPtr = reinterpret_cast<__ubuf__ float *>(yAcc.GetPhyAddr());
            SpmvAccumulateTileSegs(*this, this->indexBase, rpPtr, colPtr, valPtr, xRowPtr,
                                   yAccPtr, row, groupCount, t0, t1, winStart, winEnd);
        }
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::AccumulateWindowDs(
    uint32_t winStart, uint32_t winEnd) {
    AscendC::LocalTensor<int32_t> rowPtr = bufRowPtr.Get<int32_t>();
    AscendC::LocalTensor<float> xRow = bufXRow.Get<float>();
    AscendC::LocalTensor<int32_t> colIdx = bufColIdx.Get<int32_t>();
    AscendC::LocalTensor<float> valsF = bufValsF.Get<float>();
    AscendC::LocalTensor<float> yAcc = bufYAcc.Get<float>();
    AscendC::LocalTensor<float> yAccLo = bufYAccLo.Get<float>();
    AscendC::LocalTensor<float> redWk = bufRedWk.Get<float>();
    const float absAlphaMax = SpmvAbsF(this->alphaRe) > SpmvAbsF(this->alphaIm) ?
                              SpmvAbsF(this->alphaRe) : SpmvAbsF(this->alphaIm);

    for (uint32_t groupStart = 0; groupStart < this->blockRowNum; groupStart += SPMV_ROW_GROUP) {
        const SpmvGroupHead head = BeginGroup(groupStart);
        const uint32_t groupCount = head.groupCount;
        // 守卫外提谓词的组级量：本组 x 的 max|x 分量|（矢量归约借 bufRedWk）。
        // xRow 由 DMA 直写（LoadXGroup 仅 PIPE_MTE2 收尾，MTE2→V 实测不可靠），补全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        const float grpMaxX = SpmvVecMaxAbs(
            xRow, static_cast<int32_t>(groupCount * (INPUT_COMPLEX ? 2 : 1)), redWk);
        uint32_t row = 0;
        for (int32_t t0 = head.groupNnz0; t0 < head.groupNnz1; t0 += SPMV_TILE_ELEMS) {
            int32_t valid;
            const int32_t t1 = SpmvBeginTile(t0, head.groupNnz1, valid);
            LoadNzTile(t0, valid);

            __ubuf__ int32_t *rpPtr = reinterpret_cast<__ubuf__ int32_t *>(rowPtr.GetPhyAddr());
            __ubuf__ int32_t *colPtr = reinterpret_cast<__ubuf__ int32_t *>(colIdx.GetPhyAddr());
            __ubuf__ float *valPtr = reinterpret_cast<__ubuf__ float *>(valsF.GetPhyAddr());
            __ubuf__ float *xRowPtr = reinterpret_cast<__ubuf__ float *>(xRow.GetPhyAddr());
            __ubuf__ float *accPtr = reinterpret_cast<__ubuf__ float *>(yAcc.GetPhyAddr());
            __ubuf__ float *accLoPtr = reinterpret_cast<__ubuf__ float *>(yAccLo.GetPhyAddr());
            // tile 级谓词：tileMaxV = max|val 分量|（分块归约，tmp 上界 4096）；
            // pv = tileMaxV*grpMaxX 覆盖复乘的每个实数积，|prH| <= 2*pv 覆盖
            // alpha 结合的第二级（上界溢出为 Inf 则判不安全，方向正确）
            const uint32_t valFloats = static_cast<uint32_t>(valid) * (INPUT_COMPLEX ? 2 : 1);
            float tileMaxV = 0.0f;
            for (uint32_t off = 0; off < valFloats; off += SPMV_TILE_ELEMS) {
                uint32_t n = valFloats - off < SPMV_TILE_ELEMS ? valFloats - off : SPMV_TILE_ELEMS;
                float b = SpmvVecMaxAbs(valsF[off], static_cast<int32_t>(n), redWk);
                tileMaxV = b > tileMaxV ? b : tileMaxV;
            }
            const float pv = tileMaxV * grpMaxX;
            const bool tileSafe =
                tileMaxV <= SPMV_TP_SPLIT_SAFE_MAX && grpMaxX <= SPMV_TP_SPLIT_SAFE_MAX &&
                pv <= SPMV_TP_SPLIT_SAFE_MAX * 0.5f && absAlphaMax <= SPMV_TP_SPLIT_SAFE_MAX &&
                (pv + pv) * absAlphaMax <= SPMV_TP_PROD_SAFE_MAX;
            SpmvAccumulateTileSegsDs(*this, this->indexBase, rpPtr, colPtr, valPtr, xRowPtr,
                                     accPtr, accLoPtr, row, groupCount, t0, t1, winStart, winEnd,
                                     tileSafe, absAlphaMax);
        }
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::AccumulateSeg(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr, const __ubuf__ float *xRowPtr,
    __ubuf__ float *yAccPtr, int32_t t0, uint32_t winStart, uint32_t winEnd) {
    float xr, xi;
    SpmvFetchXRI<INPUT_COMPLEX>(xRowPtr, r, xr, xi);
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        uint32_t yIdx = 2u * static_cast<uint32_t>(c);
        if (yIdx >= winStart && yIdx < winEnd) {
            float pr, pi;
            if constexpr (INPUT_COMPLEX) {
                float vr = valPtr[2 * (j - t0)];
                float vi = valPtr[2 * (j - t0) + 1];
                pr = vr * xr - vi * xi;
                pi = vr * xi + vi * xr;
            }
            else {
                pr = valPtr[j - t0] * xr;
                pi = 0.0f;
            }
            // alpha * p（复数标量乘）后累加到窗口
            float cr = this->alphaRe * pr - this->alphaIm * pi;
            float ci = this->alphaRe * pi + this->alphaIm * pr;
            uint32_t slot = yIdx - winStart;
            yAccPtr[slot] = yAccPtr[slot] + cr;
            yAccPtr[slot + 1] = yAccPtr[slot + 1] + ci;
        }
    }
}

// 单主体实现：GUARDED 经 SpmvTwoProdSel 在逐点守卫/无守卫间编译期分派，
// 快/慢路径对正常输入逐位一致（守卫恒不动作），避免双份实现的重复块
template <bool INPUT_COMPLEX>
template <bool GUARDED>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::AccumulateSegDs(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr, const __ubuf__ float *xRowPtr,
    __ubuf__ float *accPtr, __ubuf__ float *accLoPtr, int32_t t0,
    uint32_t winStart, uint32_t winEnd) {
    float xr, xi;
    SpmvFetchXRI<INPUT_COMPLEX>(xRowPtr, r, xr, xi);
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        uint32_t yIdx = 2u * static_cast<uint32_t>(c);
        if (yIdx >= winStart && yIdx < winEnd) {
            // p = v * x（复数，Dekker 精确积分量）
            float prH, prL, piH, piL;
            if constexpr (INPUT_COMPLEX) {
                float vr = valPtr[2 * (j - t0)];
                float vi = valPtr[2 * (j - t0) + 1];
                float a, ae, b, be, t, te;
                SpmvTwoProdSel<GUARDED>(vr, xr, a, ae);
                SpmvTwoProdSel<GUARDED>(vi, xi, b, be);
                SpmvTwoSum(a, -b, t, te);
                prH = t; prL = te + (ae - be);
                SpmvTwoProdSel<GUARDED>(vr, xi, a, ae);
                SpmvTwoProdSel<GUARDED>(vi, xr, b, be);
                SpmvTwoSum(a, b, t, te);
                piH = t; piL = te + ae + be;
            }
            else {
                piH = 0.0f; piL = 0.0f;
                SpmvTwoProdSel<GUARDED>(valPtr[j - t0], xr, prH, prL);
            }
            // c = alpha * p（复数 double-single 乘）
            float a, ae, b, be, t, te;
            float crH, crL, ciH, ciL;
            SpmvTwoProdSel<GUARDED>(this->alphaRe, prH, a, ae);
            SpmvTwoProdSel<GUARDED>(this->alphaIm, piH, b, be);
            SpmvTwoSum(a, -b, t, te);
            crH = t; crL = te + (ae - be) + (this->alphaRe * prL - this->alphaIm * piL);
            SpmvTwoProdSel<GUARDED>(this->alphaRe, piH, a, ae);
            SpmvTwoProdSel<GUARDED>(this->alphaIm, prH, b, be);
            SpmvTwoSum(a, b, t, te);
            ciH = t; ciL = te + ae + be + (this->alphaRe * piL + this->alphaIm * prL);
            // TwoSum 合并入 (acc, accLo)
            uint32_t slot = yIdx - winStart;
            SpmvDsMerge(accPtr, accLoPtr, slot, crH, crL);
            SpmvDsMerge(accPtr, accLoPtr, slot + 1, ciH, ciL);
        }
    }
}


// c64 行级安全谓词：rowMax = 行段内（限窗口）max|val 分量|，xc = 本行
// max(|xr|,|xi|)；pv = rowMax*xc 为复乘每个实数积的上界，|prH|,|piH| <= 2*pv。
// 安全 ⟺ rowMax、xc 在拆分安全域、pv 在半拆分安全域（prH 拆分需 2*pv 不越限）
// 且 2*pv*|alpha|max 在积安全域。上界乘法自身溢出为 Inf 时判不安全，方向正确
template <bool INPUT_COMPLEX>
__aicore__ inline bool SpmvKernelC64Trans<INPUT_COMPLEX>::RowDsSafe(
    uint32_t r, int32_t lo, int32_t hi,
    const __ubuf__ int32_t *colPtr, const __ubuf__ float *valPtr, const __ubuf__ float *xRowPtr,
    int32_t t0, uint32_t winStart, uint32_t winEnd, float absAlphaMax) {
    float xr, xi;
    SpmvFetchXRI<INPUT_COMPLEX>(xRowPtr, r, xr, xi);
    const float ar = SpmvAbsF(xr), ai = SpmvAbsF(xi);
    const float xc = ar > ai ? ar : ai;
    float rowMax = 0.0f;
    for (int32_t j = lo; j < hi; j++) {
        int32_t c = colPtr[j - t0];
        uint32_t yIdx = 2u * static_cast<uint32_t>(c);
        if (yIdx >= winStart && yIdx < winEnd) {
            float av = SpmvAbsF(valPtr[(j - t0) * (INPUT_COMPLEX ? 2 : 1)]);
            rowMax = av > rowMax ? av : rowMax;
            if constexpr (INPUT_COMPLEX) {
                av = SpmvAbsF(valPtr[2 * (j - t0) + 1]);
                rowMax = av > rowMax ? av : rowMax;
            }
        }
    }
    const float pv = rowMax * xc;
    return rowMax <= SPMV_TP_SPLIT_SAFE_MAX && xc <= SPMV_TP_SPLIT_SAFE_MAX &&
           pv <= SPMV_TP_SPLIT_SAFE_MAX * 0.5f && absAlphaMax <= SPMV_TP_SPLIT_SAFE_MAX &&
           (pv + pv) * absAlphaMax <= SPMV_TP_PROD_SAFE_MAX;
}

// c64 finalize：按列切分，跨核 TwoSum 链合并各组 (hi, lo) 平面（分量独立，矢量），
// 再与 beta*y0 做复数 double-single 结合（标量），单次舍入写出
template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::FinalizeDs() {
    if (this->yBlockCount == 0)
        return;
    AscendC::LocalTensor<float> big = bufYAcc.Get<float>();
    AscendC::LocalTensor<float> bigLo = bufYAccLo.Get<float>();
    const uint32_t stgOff = 0;      // stgHi = big[0], stgLo = bigLo[0]
    const uint32_t accOff = 4096;   // accHi = big[4096], accLo = bigLo[4096]
    const uint32_t tmpVOff = 8192;  // tmpV = big[8192]
    const uint32_t y0Off = 8192;    // y0 staging = bigLo[8192]

    for (uint32_t colOff = 0; colOff < this->yBlockCount; colOff += SPMV_TILE_ELEMS_C64) {
        uint32_t chunkCols = SPMV_TILE_ELEMS_C64;
        if (colOff + chunkCols > this->yBlockCount)
            chunkCols = this->yBlockCount - colOff;
        const uint32_t chunkFloats = 2 * chunkCols;
        const uint32_t col0 = this->yBlockStart + colOff;

        AscendC::Duplicate(big[accOff], 0.0f, chunkFloats);
        AscendC::Duplicate(bigLo[accOff], 0.0f, chunkFloats);
        // DS finalize 起点屏障：去屏障/细粒度化实测破环（trans 路径），保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t g = 0; g < this->dsGroups; g++) {
            MergeGroupPlanes(g, chunkFloats, col0, accOff, tmpVOff);
        }

        __ubuf__ float *hiPtr = reinterpret_cast<__ubuf__ float *>(big.GetPhyAddr()) + accOff;
        __ubuf__ float *loPtr = reinterpret_cast<__ubuf__ float *>(bigLo.GetPhyAddr()) + accOff;
        __ubuf__ float *y0Ptr = reinterpret_cast<__ubuf__ float *>(bigLo.GetPhyAddr()) + y0Off;
        __ubuf__ float *outPtr = reinterpret_cast<__ubuf__ float *>(big.GetPhyAddr()) + tmpVOff;
        if (this->readY != 0) {
            AscendC::DataCopyExtParams cpY{1, (uint32_t)(chunkFloats * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padY{false, 0, 0, 0};
            AscendC::DataCopyPad(bigLo[y0Off], yVecFGm[2 * col0], cpY, padY);
            // MTE2→标量读 y0Ptr：细粒度化实测破环（c64 trans DS finalize），保留全屏障
            AscendC::PipeBarrier<PIPE_ALL>();
            CombineBetaY0(chunkCols, hiPtr, loPtr, y0Ptr, outPtr);
        }
        else {
            for (uint32_t j = 0; j < chunkFloats; j++) {
                outPtr[j] = hiPtr[j] + loPtr[j];
            }
        }
        // 标量写 outPtr→MTE3 写出：无细粒度屏障可用，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cpO{1, (uint32_t)(chunkFloats * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yVecFGm[2 * col0], big[tmpVOff], cpO);
        // MTE3 读出→次 chunk 覆写 WAR，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::MergeGroupPlanes(
    uint32_t g, uint32_t chunkFloats, uint32_t col0, uint32_t accOff, uint32_t tmpVOff) {
    AscendC::LocalTensor<float> big = bufYAcc.Get<float>();
    AscendC::LocalTensor<float> bigLo = bufYAccLo.Get<float>();
    AscendC::LocalTensor<float> tmpT = bufColIdx.Get<float>();
    AscendC::LocalTensor<float> tmpU = bufValsF.Get<float>();
    uint64_t hiBase = static_cast<uint64_t>(g) * this->dsStride + 2 * col0;
    uint64_t loBase = static_cast<uint64_t>(this->dsGroups + g) * this->dsStride + 2 * col0;
    AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkFloats * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(big[0], wsGm[hiBase], cp, pad);
    AscendC::DataCopyPad(bigLo[0], wsGm[loBase], cp, pad);
    // MTE2（staging 入 UB）→矢量 TwoSum：实测 PipeBarrier<PIPE_MTE2> 对 MTE2→V 不可靠（arch22），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Add(tmpT, big[accOff], big[0], chunkFloats);   // t = a + b
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpU, tmpT, big[accOff], chunkFloats);     // bp = t - a
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(big[tmpVOff], tmpT, tmpU, chunkFloats);    // t - bp
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(big[tmpVOff], big[accOff], big[tmpVOff], chunkFloats); // a - (t - bp)
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(tmpU, big[0], tmpU, chunkFloats);          // b - bp
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(big[tmpVOff], big[tmpVOff], tmpU, chunkFloats); // e
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(tmpU, bigLo[0], bigLo[accOff], chunkFloats);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(bigLo[accOff], big[tmpVOff], tmpU, chunkFloats);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(big[accOff], tmpT, 0.0f, chunkFloats);
    // DS 合并链收尾屏障：细粒度化实测破环（trans 路径），保留全屏障
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::CombineBetaY0(
    uint32_t chunkCols, const __ubuf__ float *hiPtr, const __ubuf__ float *loPtr,
    const __ubuf__ float *y0Ptr, __ubuf__ float *outPtr) {
    for (uint32_t j = 0; j < chunkCols; j++) {
        // beta*y0 复数 double-single 乘
        float a, ae, b, be, t, te;
        float brH, brL, biH, biL;
        SpmvTwoProd(this->betaRe, y0Ptr[2 * j], a, ae);
        SpmvTwoProd(this->betaIm, y0Ptr[2 * j + 1], b, be);
        SpmvTwoSum(a, -b, t, te);
        brH = t; brL = te + (ae - be);
        SpmvTwoProd(this->betaRe, y0Ptr[2 * j + 1], a, ae);
        SpmvTwoProd(this->betaIm, y0Ptr[2 * j], b, be);
        SpmvTwoSum(a, b, t, te);
        biH = t; biL = te + ae + be;
        // + acc，单次舍入
        float s, e;
        SpmvTwoSum(hiPtr[2 * j], brH, s, e);
        outPtr[2 * j] = s + (loPtr[2 * j] + brL + e);
        SpmvTwoSum(hiPtr[2 * j + 1], biH, s, e);
        outPtr[2 * j + 1] = s + (loPtr[2 * j + 1] + biL + e);
    }
}

template <bool INPUT_COMPLEX>
__aicore__ inline void SpmvKernelC64Trans<INPUT_COMPLEX>::Process() {
    // double-single 路径：不做 beta 预缩放与 GM 原子加（beta 在 finalize 里
    // double-single 结合；每核写自己专属的 ws 平面，跨核顺序无关）
    AscendC::LocalTensor<float> yAcc = bufYAcc.Get<float>();
    AscendC::LocalTensor<float> yAccLo = bufYAccLo.Get<float>();
    const uint32_t yFloats = this->totalColsNum * 2;
    const uint32_t winCountTotal = (yFloats + YWIN_FLOATS - 1) / YWIN_FLOATS;

    for (uint32_t w = 0; w < winCountTotal; w++) {
        const uint32_t winStart = w * YWIN_FLOATS;
        uint32_t winFloats = yFloats - winStart;
        if (winFloats > YWIN_FLOATS)
            winFloats = YWIN_FLOATS;

        AscendC::Duplicate(yAcc, 0.0f, winFloats);
        AscendC::Duplicate(yAccLo, 0.0f, winFloats);
        // 矢量清零 yAcc/yAccLo → 标量行段累加读改写
        AscendC::PipeBarrier<PIPE_V>();
        if (this->blockRowNum > 0) {
            AccumulateWindowDs(winStart, winStart + winFloats);
        }

        // 本核专属平面普通写出 (acc, accLo)；空载核（行数少于核数）也写零平面
        // 标量写 acc→MTE3 写出：无细粒度屏障可用，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
        if (AscendC::GetBlockIdx() < this->dsGroups) {
            uint64_t hiBase = static_cast<uint64_t>(AscendC::GetBlockIdx()) * this->dsStride + winStart;
            uint64_t loBase = static_cast<uint64_t>(this->dsGroups + AscendC::GetBlockIdx()) * this->dsStride + winStart;
            AscendC::DataCopyExtParams cp{1, (uint32_t)(winFloats * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(wsGm[hiBase], yAcc, cp);
            AscendC::DataCopyPad(wsGm[loBase], yAccLo, cp);
        }
        // MTE3 读 yAcc→次窗口矢量清零覆写 WAR，保留全屏障
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll();
    FinalizeDs();
}

// -------------------------------------------------------------------
// 具体 Kernel 入口 + Host 端启动封装
// -------------------------------------------------------------------

// ---- per-type __global__ kernel 定义（各自 TU 展开一个实例） ----
#define DEFINE_SPMV_KERNEL(CompT, ValT, OutT)                                                           \
    __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT(                                   \
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,                                           \
        GM_ADDR xVec, GM_ADDR yVec, GM_ADDR ws,                                                          \
        uint32_t totalRowsNum, uint32_t totalColNum,                                                     \
        CompT alpha, CompT beta, bool trans, uint32_t indexBase, uint32_t readY)                        \
    {                                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                 \
        if (trans)                                                                                      \
        {                                                                                               \
            SpmvKernelTrans<CompT, ValT, OutT> op;                                                      \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, ws, totalRowsNum, totalColNum,            \
                    alpha, beta, indexBase, readY);                                                     \
            op.Process();                                                                               \
        }                                                                                               \
        else                                                                                            \
        {                                                                                               \
            SpmvKernel<CompT, ValT, OutT> op;                                                           \
            (void)ws;                                                                                   \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum,                \
                    alpha, beta, indexBase, readY);                                                     \
            op.Process();                                                                               \
        }                                                                                               \
    }

// complex64 kernel：alpha/beta 拆为实/虚部标量传入；INPUT_COMPLEX 区分输入是否复数
#define DEFINE_SPMV_KERNEL_C64(NAME, INPUT_COMPLEX)                                                      \
    __global__ __aicore__ void spmv_kernel_c64_##NAME(                                                   \
        GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal,                                            \
        GM_ADDR xVec, GM_ADDR yVec, GM_ADDR ws,                                                           \
        uint32_t totalRowsNum, uint32_t totalColNum,                                                      \
        float alphaRe, float alphaIm, float betaRe, float betaIm,                                         \
        bool trans, uint32_t indexBase, uint32_t readY)                                                   \
    {                                                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                   \
        if (trans)                                                                                        \
        {                                                                                                 \
            SpmvKernelC64Trans<INPUT_COMPLEX> op;                                                         \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, ws, totalRowsNum, totalColNum,              \
                    alphaRe, alphaIm, betaRe, betaIm, indexBase, readY);                                  \
            op.Process();                                                                                 \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            SpmvKernelC64<INPUT_COMPLEX> op;                                                              \
            (void)ws;                                                                                     \
            op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum, totalColNum,                  \
                    alphaRe, alphaIm, betaRe, betaIm, indexBase, readY);                                  \
            op.Process();                                                                                 \
        }                                                                                                 \
    }

// ---- extern 声明（供 spmv_kernel_do 引用各 TU 中的 __global__ kernel） ----
// 注意：f32_i8_f32 / i32_i8_i32 / c64_c64 / c64_f32 四种组合的定义已并入
// spmv_kernel.cpp 同一 TU，此处不再重复声明（同 TU 内 extern 声明与 DEFINE
// 展开共存会在 host 编译 pass 形成二义性重载）。
#define SPMV_EXTERN(CompT, ValT, OutT) \
    extern __global__ __aicore__ void spmv_kernel_##CompT##_##ValT##_##OutT( \
        GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, CompT, CompT, bool, uint32_t, uint32_t);
SPMV_EXTERN(float, float, float)
SPMV_EXTERN(float, half, half)
SPMV_EXTERN(float, half, float)
SPMV_EXTERN(float, bfloat16_t, bfloat16_t)
SPMV_EXTERN(float, bfloat16_t, float)
SPMV_EXTERN(float, int32_t, float)
SPMV_EXTERN(int32_t, int32_t, int32_t)
#undef SPMV_EXTERN
