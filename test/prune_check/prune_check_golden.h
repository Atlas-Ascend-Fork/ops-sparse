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

#ifndef TEST_PRUNE_CHECK_GOLDEN_H_
#define TEST_PRUNE_CHECK_GOLDEN_H_

// =============================================================================
// CPU Golden reference + 数据构造器 for aclsparseLtSpMMAPruneCheck.
//
// 数学定义（需求文档 §1 / 2.1 测试方案 §2.1）：
//   nnz(g) = #{x ∈ g : x != 0}（位级判零，见下）
//   d_valid = 0，若 ∀ g: nnz(g) ≤ keep；d_valid = 1，若 ∃ g: nnz(g) > keep
//   批量扩展：d_valid = OR(全部 batch 的判定)（PruneCheckGoldenBatch，单 batch
//   判定逻辑不变）
//   分组规则：FP32 → group=2, keep=1（1:2）；FP16/BF16/INT8 → group=4, keep=2（2:4）
//   剪枝方向：alongRow = (trans != isRowOrder)（与 prune golden SpMMAPruneGolden
//             完全同源）：沿行 = N+ROW / T+COL；沿列 = N+COL / T+ROW
//
// 判零语义（位级实现）：
//   (bits & magMask) != 0，magMask = FP32:0x7fffffff / FP16,BF16:0x7fff / INT8:0xff。
//   - +0.0 / -0.0（幅值位全零）→ 零；
//   - NaN / ±inf / subnormal（幅值位非零）→ 计为非零元素；
//   - INT8 0x00 → 零；±127 / -128 → 非零。
//   与 IEEE 相等语义 x != 0 逐位等价，全程无浮点转换链（对齐 kernel 位视图实现，
//   规避 BF16 cast 后端陷阱，开发方案 §2/§6.1 R8）。
//
// 尾部不完整组（防御性保留）：描述符层强制 rows/cols/ld 为 32B/elem
// 元素倍数（FP32:8 / FP16,BF16:16 / INT8:32），分组维度恒为 g 的倍数，尾部
// 不完整组在描述符可达空间内不可达（相关 shape 用例为描述符层拦截断言）。
// golden 仍按「尾部组以实际元素数与同一 keep 比较」实现，供独立编译自检
// 与未来约束放宽时复用。
//
// 本头文件为纯 CPU 参考，不依赖任何 aclsparse / ACL API，可独立编译
// （范式对齐 test/prune/prune_golden.h）；PRUNED_SIM 构造器复用 prune golden
// 的 SpMMAPruneGolden（CPU 仿真剪枝产物必过 check，官方 "guaranteed" 性质）。
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// 复用 prune 家族已验证的 dtype 存储类型与转换（bf16_bits_t 等）及 CPU prune
// 仿真（SpMMAPruneGolden，PRUNED_SIM 构造器输入源）。
#include "prune_golden.h"
#include "prune_test_util.h"

namespace sparse_test {

// -----------------------------------------------------------------------------
// 免不安全字节拷贝函数的等宽位重解释（codecheck 按 C 内存拷贝类内建函数
// 报告，须规避；C++17 无 std::bit_cast，以 char 逐字节循环实现，char 可
// 安全别名任意对象表示，编译器可将其完全优化为单次 load/store）。
// -----------------------------------------------------------------------------
template <typename Dst, typename Src>
inline Dst BitCastBits(const Src& value)
{
    static_assert(sizeof(Dst) == sizeof(Src), "bit-cast requires equal object sizes");
    Dst result;
    char* dst = reinterpret_cast<char*>(&result);
    const char* src = reinterpret_cast<const char*>(&value);
    for (size_t i = 0; i < sizeof(Src); ++i) {
        dst[i] = src[i];
    }
    return result;
}

// -----------------------------------------------------------------------------
// 位级判零 trait：IsNonzeroBits<T>(v) === (存储位模式 & magMask) != 0。
// -----------------------------------------------------------------------------
template <typename T>
struct PruneCheckBitsTrait;

template <>
struct PruneCheckBitsTrait<float> {
    static bool nonzero(float v) {
        const uint32_t bits = BitCastBits<uint32_t>(v);
        return (bits & 0x7fffffffU) != 0U;  // ±0 → 零；NaN/±inf/subnormal → 非零
    }
};

template <>
struct PruneCheckBitsTrait<uint16_t> {  // FP16 位存储
    static bool nonzero(uint16_t v) { return (v & 0x7fffU) != 0U; }
};

template <>
struct PruneCheckBitsTrait<bf16_bits_t> {  // BF16 位存储
    static bool nonzero(bf16_bits_t v) { return (v.value & 0x7fffU) != 0U; }
};

template <>
struct PruneCheckBitsTrait<int8_t> {
    static bool nonzero(int8_t v) {
        return (static_cast<uint8_t>(v) & 0xffU) != 0U;  // 0x00 → 零；其余非零
    }
};

// -----------------------------------------------------------------------------
// 特殊值编码（S1-S6 特殊值场景的位模式来源；INT8 无 IEEE 特殊值，见各分支）。
// -----------------------------------------------------------------------------
enum class SpecialVal : int {
    kNormal = 0,     // 常规非零值（均匀随机 [lo, hi)，保证位模式非零）
    kNegZero,        // -0.0 位模式（FP16 0x8000 / BF16 0x8000 / FP32 0x80000000）
    kSubnormal,      // 最小非规格数（FP16 0x0001 / BF16 0x0001 / FP32 0x00000001）
    kNan,            // quiet NaN（FP16 0x7e00 / BF16 0x7fc0 / FP32 0x7fc00000）
    kPosInf,         // +inf（FP16 0x7c00 / BF16 0x7f80 / FP32 0x7f800000）
};

template <typename T>
struct SpecialValCodec;

template <>
struct SpecialValCodec<float> {
    static float bits(uint32_t b) { return BitCastBits<float>(b); }
    static float make(SpecialVal sv) {
        switch (sv) {
            case SpecialVal::kNegZero:   return bits(0x80000000U);
            case SpecialVal::kSubnormal: return bits(0x00000001U);
            case SpecialVal::kNan:       return bits(0x7fc00000U);
            case SpecialVal::kPosInf:    return bits(0x7f800000U);
            default:                     return 0.0f;
        }
    }
};

template <>
struct SpecialValCodec<uint16_t> {  // FP16
    static uint16_t make(SpecialVal sv) {
        switch (sv) {
            case SpecialVal::kNegZero:   return 0x8000U;
            case SpecialVal::kSubnormal: return 0x0001U;
            case SpecialVal::kNan:       return 0x7e00U;
            case SpecialVal::kPosInf:    return 0x7c00U;
            default:                     return 0x0000U;
        }
    }
};

template <>
struct SpecialValCodec<bf16_bits_t> {
    static bf16_bits_t make(SpecialVal sv) {
        switch (sv) {
            case SpecialVal::kNegZero:   return bf16_bits_t(0x8000U);
            case SpecialVal::kSubnormal: return bf16_bits_t(0x0001U);
            case SpecialVal::kNan:       return bf16_bits_t(0x7fc0U);
            case SpecialVal::kPosInf:    return bf16_bits_t(0x7f80U);
            default:                     return bf16_bits_t(0x0000U);
        }
    }
};

template <>
struct SpecialValCodec<int8_t> {
    // INT8 无 IEEE 特殊值：-0 退化为 0x00（判零）；subnormal/NAN/INF 退化为
    // 非零边界值 0x01 / 0x7f / 0x80（值域极值语义，S6）。
    static int8_t make(SpecialVal sv) {
        switch (sv) {
            case SpecialVal::kNegZero:   return static_cast<int8_t>(0x00);
            case SpecialVal::kSubnormal: return static_cast<int8_t>(0x01);
            case SpecialVal::kNan:       return static_cast<int8_t>(0x7f);
            case SpecialVal::kPosInf:    return static_cast<int8_t>(0x80);
            default:                     return static_cast<int8_t>(0);
        }
    }
};

// -----------------------------------------------------------------------------
// 分组几何遍历（方向推导与 golden / kernel 共享）。
// A 为逻辑 (m, k) 行主序；调用方对每组收到元素坐标步进 (dRow, dCol)：
//   alongRow = true  : 每行内连续 g 个元素一组（组内 dRow=0, dCol=1）
//   alongRow = false : 每列内连续 g 行一组（组内 dRow=1, dCol=0）
// fn(rowStart, colStart, dRow, dCol, count)：count 为该组实际元素数（尾部
// 不完整组 < g，防御性保留语义）。
// -----------------------------------------------------------------------------
template <typename F>
inline void ForEachPruneGroup(int32_t m, int32_t k, bool alongRow, int32_t group, F&& fn) {
    if (alongRow) {
        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < k; j += group) {
                const int32_t cnt = std::min(group, k - j);
                fn(i, j, 0, 1, cnt);
            }
        }
    } else {
        for (int32_t j = 0; j < k; ++j) {
            for (int32_t i = 0; i < m; i += group) {
                const int32_t cnt = std::min(group, m - i);
                fn(i, j, 1, 0, cnt);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Golden 入口：校验逻辑 (m, k) 行主序矩阵 A 是否满足 2:4（FP32 1:2）剪枝条件。
//
// 模板参数 T ∈ {float, uint16_t(FP16 bits), bf16_bits_t, int8_t}。
// trans / isRowOrder 为稀疏侧 op / order；alongRow = (trans != isRowOrder)。
// 返回 0（全部组 nnz ≤ keep）/ 1（存在组 nnz > keep）。
// 与 kernel 的对应：kernel 在物理行主序 (ld 步长) 上扫描（T+ROW 时物理为转置
// (k,m)），两种几何在本逻辑空间定义下逐组等价（方向推导同源保证）。
// -----------------------------------------------------------------------------
template <typename T>
int PruneCheckGolden(const std::vector<T>& A, int32_t m, int32_t k,
                     bool trans, bool isRowOrder)
{
    const int group = std::is_same_v<T, float> ? 2 : 4;
    const int keep = group / 2;
    const bool alongRow = (trans != isRowOrder);

    int valid = 0;
    ForEachPruneGroup(m, k, alongRow, group, [&](int32_t r0, int32_t c0,
                                                 int32_t dR, int32_t dC, int32_t cnt) {
        int nnz = 0;
        for (int t = 0; t < cnt; ++t) {
            const size_t idx = static_cast<size_t>(r0 + t * dR) * static_cast<size_t>(k) +
                               static_cast<size_t>(c0 + t * dC);
            if (PruneCheckBitsTrait<T>::nonzero(A[idx])) { ++nnz; }
        }
        // 尾部不完整组按实际元素数与同一 keep 比较（防御性，见文件头注释）。
        if (nnz > keep) { valid = 1; }
    });
    return valid;
}

// -----------------------------------------------------------------------------
// 批量 Golden：d_valid = OR(全部 batch 的判定)。物理批量缓冲按 batchStride
// 切分（batch b 基址 = b×batchStride 元素），逐 batch 提取逻辑 (m, k) 行主序
// 矩阵后复用单 batch 判定（位级判零语义不变）。物理布局与 SparseSideSpecOf
// 一致：仅 T+ROW 为转置 (k, m) 行主序（ld 步长），其余组合物理即逻辑 (m, k)
// 行主序；ld 为物理行步长。
// ld padding 语义（与 kernel srcStride 消费点对齐）：
// 提取仅取数据区 [0, 物理行长)（非转置 j∈[0,k)、转置 i∈[0,m)），padding 区
// [物理行长, ld) 不进入分组判定 —— padding ld 用例（PruneCheckPaddingLdTest）
// 以「逻辑 golden == 物理 ld golden」双口径交叉断言固化该语义；垃圾填充由
// wrapper 的 PrepareSparseSideData 承载（PruneCheckPaddingGarbage），golden
// 侧无需扩展。
// -----------------------------------------------------------------------------
template <typename T>
int PruneCheckGoldenBatch(const std::vector<T>& hPhys, int32_t m, int32_t k,
                          bool trans, bool isRowOrder, int32_t numBatches,
                          int64_t batchStride, int64_t ld)
{
    // 仅 T+ROW 的物理存储为转置（logical[i][j] = phys[j*ld + i]）。
    const bool transposeLayout = trans && isRowOrder;
    int valid = 0;
    std::vector<T> batch(static_cast<size_t>(m) * static_cast<size_t>(k));
    for (int32_t b = 0; b < numBatches; ++b) {
        const size_t base = static_cast<size_t>(b) * static_cast<size_t>(batchStride);
        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < k; ++j) {
                const size_t idx = transposeLayout
                    ? base + static_cast<size_t>(j) * static_cast<size_t>(ld) + static_cast<size_t>(i)
                    : base + static_cast<size_t>(i) * static_cast<size_t>(ld) + static_cast<size_t>(j);
                batch[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] =
                    hPhys[idx];
            }
        }
        valid |= PruneCheckGolden<T>(batch, m, k, trans, isRowOrder);
    }
    return valid;
}

// -----------------------------------------------------------------------------
// 数据构造器（与 golden 共享方向推导 / 判零语义；全部种子化确定性）。
//
// pattern（2.1 §2.1 构造器表）           | 产出预期 d_valid
// ---------------------------------------+------------------
// VALID_SPARSE   每组恰 keep 非零（合法上边界；尾组 min(keep, count)）   | 0
// VALID_MIXED    每组 nnz ∈ [0, keep] 随机（含全零组与不足组）           | 0
// ALL_ZERO       全零（nnz 下边界）                                     | 0
// SINGLE_VIOLATION 指定位置组 nnz = keep+1（viol_pos，其余组合法）      | 1
// MULTI_VIOLATION  首/中/尾三组各一处违规                               | 1
// ALL_VIOLATING  每组 nnz = keep+1（组元素数不足时取 count，见防御说明） | 1
// RANDOM_DENSE   值域均匀随机稠密（统计必然存在违规组）                 | 1
// PRUNED_SIM_STRIP / _TILE  随机矩阵经 SpMMAPruneGolden CPU 仿真        | 0
// VALID_NEGZERO  VALID_SPARSE + 零位填 -0.0（±0 判零 → nnz 不变）       | 0
// VALID_SUBNORMAL VALID_SPARSE + 非零位取最小非规格数（计非零）         | 0
// NAN_VALID      VALID_SPARSE + 非零位取 NaN（计非零 → 恰 keep）        | 0
// NAN_VIOLATING  SINGLE_VIOLATION + 违规组非零取 NaN（keep+1）          | 1
// INF_VIOLATING  SINGLE_VIOLATION + 违规组非零取 ±inf（keep+1）         | 1
//
// viol_pos：FIRST / MIDDLE / LAST / MULTI；LAST 在存在尾部不完整组时优先指向
// 尾组（尾组元素数不足以违规时回退到最后一个可违规组，防御性——可达空间内
// 无尾组，该回退不触发）。
// -----------------------------------------------------------------------------

enum class ViolPosKind { kNone, kFirst, kMiddle, kLast, kMulti };

inline ViolPosKind ParseViolPos(const std::string& s) {
    if (s == "FIRST") { return ViolPosKind::kFirst; }
    if (s == "MIDDLE") { return ViolPosKind::kMiddle; }
    if (s == "LAST") { return ViolPosKind::kLast; }
    if (s == "MULTI") { return ViolPosKind::kMulti; }
    return ViolPosKind::kNone;
}

namespace prune_check_detail {

// 组描述：组首元素坐标 + 步进 + 实际元素数。
struct GroupRef {
    int32_t r0, c0, dR, dC, cnt;
    size_t elem(int32_t t, int32_t k) const {
        return static_cast<size_t>(r0 + t * dR) * static_cast<size_t>(k) +
               static_cast<size_t>(c0 + t * dC);
    }
};

inline std::vector<GroupRef> CollectGroups(int32_t m, int32_t k, bool alongRow, int32_t group) {
    std::vector<GroupRef> groups;
    ForEachPruneGroup(m, k, alongRow, group, [&](int32_t r0, int32_t c0,
                                                 int32_t dR, int32_t dC, int32_t cnt) {
        groups.push_back(GroupRef{r0, c0, dR, dC, cnt});
    });
    return groups;
}

// 从组内 count 个位置中确定性抽取 n 个不同位置（Fisher-Yates 前 n 步）。
inline std::vector<int32_t> PickPositions(int32_t count, int32_t n, std::mt19937& rng) {
    std::vector<int32_t> idx(count);
    for (int32_t i = 0; i < count; ++i) { idx[i] = i; }
    const int32_t take = std::min(n, count);
    for (int32_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<int32_t> dist(i, count - 1);
        std::swap(idx[i], idx[dist(rng)]);
    }
    idx.resize(take);
    return idx;
}

// 常规非零值：均匀随机 [lo, hi)，并保证位模式非零（防 lo/hi 退化产生 ±0）。
template <typename T>
T MakeNormalNonzero(float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int attempt = 0; attempt < 8; ++attempt) {
        const float f = dist(rng);
        T v;
        if constexpr (std::is_same_v<T, float>) {
            v = f;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            v = Fp32ToFp16Bits(f);
        } else if constexpr (std::is_same_v<T, bf16_bits_t>) {
            v = bf16_bits_t(Fp32ToBf16Bits(f));
        } else {
            int32_t iv = static_cast<int32_t>(std::round(f));
            if (iv > 127) { iv = 127; }
            if (iv < -128) { iv = -128; }
            v = static_cast<int8_t>(iv);
        }
        if (PruneCheckBitsTrait<T>::nonzero(v)) { return v; }
    }
    // 兜底：最小非零位模式。
    return SpecialValCodec<T>::make(SpecialVal::kSubnormal);
}

// 把一个组填为 nnzTarget 个非零（位置随机），其余位置填 zeroFill。
template <typename T>
void FillGroup(std::vector<T>& A, int32_t k, const GroupRef& g, int32_t nnzTarget,
               SpecialVal nzVal, float lo, float hi, std::mt19937& rng) {
    const int32_t take = std::min(nnzTarget, g.cnt);
    const auto pos = PickPositions(g.cnt, take, rng);
    std::vector<bool> nz(static_cast<size_t>(g.cnt), false);
    for (int32_t p : pos) { nz[static_cast<size_t>(p)] = true; }
    for (int32_t t = 0; t < g.cnt; ++t) {
        A[g.elem(t, k)] = nz[static_cast<size_t>(t)]
            ? ((nzVal == SpecialVal::kNormal) ? MakeNormalNonzero<T>(lo, hi, rng)
                                              : SpecialValCodec<T>::make(nzVal))
            : SpecialValCodec<T>::make(SpecialVal::kNegZero);  // 存储零（-0 位模式）
    }
}
// 违规位置 → 组下标（0 基）；preferTail = LAST 且存在可违规尾组时指向尾组，
// 否则回退到最后一个可违规组（防御性，可达空间内不触发）。
inline int32_t ViolationGroupIndex(const std::vector<GroupRef>& groups, ViolPosKind vp,
                                    int32_t keep)
{
    if (groups.empty()) { return -1; }
    const int32_t need = keep + 1;  // 可违规所需最小元素数
    int32_t firstViolable = -1, lastViolable = -1;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].cnt >= need) {
            if (firstViolable < 0) { firstViolable = static_cast<int32_t>(i); }
            lastViolable = static_cast<int32_t>(i);
        }
    }
    if (firstViolable < 0) { return -1; }  // 任何组都无法违规（防御性）
    switch (vp) {
        case ViolPosKind::kFirst:  return firstViolable;
        case ViolPosKind::kLast:   return lastViolable;
        case ViolPosKind::kMiddle: return (firstViolable + lastViolable) / 2;
        default:                   return firstViolable;
    }
}

// ----- 构造器拆分（超大函数收敛；rng 逐次透传，消耗序与拆分前完全一致） -----

// 整矩阵类 pattern（ALL_ZERO / PRUNED_SIM_* / RANDOM_DENSE）：不经分组构造，
// 直接产出整矩阵；handled=false 表示非本类 pattern（交由组级构造）。
template <typename T>
std::vector<T> GenWholeMatrixInput(const std::string& pattern, int32_t m, int32_t k,
                                   bool alongRow, float lo, float hi, uint32_t seed,
                                   bool& handled)
{
    handled = true;
    if (pattern == "ALL_ZERO") {
        return std::vector<T>(static_cast<size_t>(m) * static_cast<size_t>(k),
                              SpecialValCodec<T>::make(SpecialVal::kNegZero));
    }
    if (pattern != "PRUNED_SIM_STRIP" && pattern != "PRUNED_SIM_TILE") {
        if (pattern == "RANDOM_DENSE") {
            return MatmulGenTestMatrix<T>(m, k, lo, hi, seed);
        }
        handled = false;
        return {};
    }
    // PRUNED_SIM：随机稠密 → CPU prune 仿真 → 产物必过 check（官方
    // "guaranteed" 性质；同时是方向推导正确性的最强交叉验证——方向不一致
    // 时合法输入会被误判违规）。
    std::vector<T> dense = MatmulGenTestMatrix<T>(m, k, lo, hi, seed);
    std::vector<T> pruned;
    const std::string pruneAlg = (pattern == "PRUNED_SIM_TILE") ? "TILE" : "STRIP";
    // SpMMAPruneGolden 方向推导与本 golden 同源：alongRow = (trans != isRowOrder)。
    // 此处以 (trans, isRowOrder) = (alongRow, false) 等价传入（false != alongRow
    // == alongRow），保证 prune 剪枝方向与 check 校验方向一致。
    SpMMAPruneGolden<T>(dense, pruned, m, k, 0, 1, alongRow, false, pruneAlg);
    return pruned;
}

// 组级 pattern 的特殊值与基型分类（值域替换 + 基型判定，语义见文件头构造器表）。
struct GroupPatternInfo {
    SpecialVal baseVal = SpecialVal::kNormal;  // 合法基型的非零值域
    SpecialVal violVal = SpecialVal::kNormal;  // 违规组的非零值域
    bool validSparseBase = false;  // VALID_SPARSE 基型（含特殊值替换变体）
    bool singleViol = false;       // 单点违规基型（含特殊值替换变体）
    bool multiViol = false;
    bool allViolating = false;
};

inline GroupPatternInfo ClassifyGroupPattern(const std::string& pattern) {
    GroupPatternInfo info;
    if (pattern == "VALID_SUBNORMAL") { info.baseVal = SpecialVal::kSubnormal; }
    if (pattern == "NAN_VALID") { info.baseVal = SpecialVal::kNan; }
    if (pattern == "NAN_VIOLATING") { info.violVal = SpecialVal::kNan; }
    if (pattern == "INF_VIOLATING") { info.violVal = SpecialVal::kPosInf; }
    info.validSparseBase = (pattern == "VALID_SPARSE" || pattern == "VALID_NEGZERO" ||
                            pattern == "VALID_SUBNORMAL" || pattern == "NAN_VALID");
    info.singleViol = (pattern == "SINGLE_VIOLATION" || pattern == "NAN_VIOLATING" ||
                       pattern == "INF_VIOLATING");
    info.multiViol = (pattern == "MULTI_VIOLATION");
    info.allViolating = (pattern == "ALL_VIOLATING");
    return info;
}

// 基线填充：全部组填为 nnzTarget 个非零（值域 nzVal），其余位存 -0。
template <typename T>
void FillAllGroups(std::vector<T>& A, const std::vector<GroupRef>& groups, int32_t k,
                   int32_t nnzTarget, SpecialVal nzVal, float lo, float hi,
                   std::mt19937& rng)
{
    for (const auto& g : groups) {
        FillGroup<T>(A, k, g, nnzTarget, nzVal, lo, hi, rng);
    }
}

// 单点违规：基线合法填充后，violPos 指定组 nnz = keep+1（值域 violVal）。
// violPos 空/未知缺省 MIDDLE（CSV 列语义，见构造器入口注释）。
template <typename T>
void ApplySingleViolation(std::vector<T>& A, const std::vector<GroupRef>& groups,
                          int32_t k, const std::string& violPos,
                          int32_t keep, SpecialVal violVal, float lo, float hi,
                          std::mt19937& rng)
{
    ViolPosKind vp = ParseViolPos(violPos);
    if (vp == ViolPosKind::kNone) { vp = ViolPosKind::kMiddle; }
    const int32_t gi = ViolationGroupIndex(groups, vp, keep);
    if (gi >= 0) {
        FillGroup<T>(A, k, groups[static_cast<size_t>(gi)], keep + 1, violVal, lo, hi, rng);
    }
}

// 多点违规：首/中/尾三组各一处违规（跨核归约位置敏感性）。
template <typename T>
void ApplyMultiViolation(std::vector<T>& A, const std::vector<GroupRef>& groups,
                          int32_t k, int32_t keep, float lo, float hi,
                          std::mt19937& rng)
{
    for (ViolPosKind vp : {ViolPosKind::kFirst, ViolPosKind::kMiddle, ViolPosKind::kLast}) {
        const int32_t gi = ViolationGroupIndex(groups, vp, keep);
        if (gi >= 0) {
            FillGroup<T>(A, k, groups[static_cast<size_t>(gi)], keep + 1,
                         SpecialVal::kNormal, lo, hi, rng);
        }
    }
}

// 组级构造（VALID_* / *VIOLATION* / VALID_MIXED）：按分类派发填充。
template <typename T>
std::vector<T> GenGroupBasedInput(const std::string& pattern, int32_t m, int32_t k,
                                  bool alongRow, const std::string& violPos,
                                  float lo, float hi, std::mt19937& rng)
{
    const int group = std::is_same_v<T, float> ? 2 : 4;
    const int keep = group / 2;
    const size_t mk = static_cast<size_t>(m) * static_cast<size_t>(k);
    const auto groups = CollectGroups(m, k, alongRow, group);
    const GroupPatternInfo info = ClassifyGroupPattern(pattern);
    if (!info.validSparseBase && !info.singleViol && !info.multiViol && !info.allViolating &&
        pattern != "VALID_MIXED") {
        // 未知 pattern：交由调用方 fail（测试侧显式断言 pattern 合法性）。
        return std::vector<T>{};
    }

    // 零位填充值：VALID_NEGZERO 显式用 -0.0 位模式（S1：±0 判零 → nnz 不变）。
    std::vector<T> A(mk);
    for (size_t i = 0; i < mk; ++i) { A[i] = SpecialValCodec<T>::make(SpecialVal::kNegZero); }

    if (info.validSparseBase) {
        // 每组恰 keep 非零（合法上边界）；尾组 min(keep, cnt)。
        FillAllGroups<T>(A, groups, k, keep, info.baseVal, lo, hi, rng);
        return A;
    }
    if (pattern == "VALID_MIXED") {
        // 每组 nnz ∈ [0, keep] 随机（含全零组与不足组）。
        for (const auto& g : groups) {
            std::uniform_int_distribution<int32_t> dist(0, keep);
            FillGroup<T>(A, k, g, dist(rng), SpecialVal::kNormal, lo, hi, rng);
        }
        return A;
    }
    if (info.allViolating) {
        // 每组 nnz = keep+1（组元素数不足时取 count —— 该组 nnz ≤ keep 不构成
        // 违规，防御性；可达空间内分组维度恒为 g 倍数，不触发）。
        FillAllGroups<T>(A, groups, k, keep + 1, SpecialVal::kNormal, lo, hi, rng);
        return A;
    }
    // 单点 / 多点违规：其余组保持合法（恰 keep），目标组 nnz = keep+1。
    FillAllGroups<T>(A, groups, k, keep, SpecialVal::kNormal, lo, hi, rng);
    if (info.singleViol) {
        ApplySingleViolation<T>(A, groups, k, violPos, keep, info.violVal, lo, hi,
                                rng);
    } else {
        ApplyMultiViolation<T>(A, groups, k, keep, lo, hi, rng);
    }
    return A;
}

}  // namespace prune_check_detail

// -----------------------------------------------------------------------------
// 构造器入口：按 pattern 生成逻辑 (m, k) 行主序待校验矩阵。
// violPos 为 CSV viol_pos 列（FIRST / MIDDLE / LAST / MULTI / 空），仅违规类
// pattern 有效；空缺省 MIDDLE。沿用 prune 测试的确定性种子约定（可复现）。
// 实现拆分见 prune_check_detail：整矩阵类 pattern（ALL_ZERO / PRUNED_SIM /
// RANDOM_DENSE）不经分组直接产出，其余走组级构造；rng 逐次透传，消耗序与
// 拆分前完全一致（种子化确定性不变）。
// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> GenPruneCheckInput(const std::string& pattern, int32_t m, int32_t k,
                                  bool alongRow, const std::string& violPos,
                                  float lo, float hi, uint32_t seed)
{
    using namespace prune_check_detail;
    std::mt19937 rng(seed);
    bool wholeMatrix = false;
    std::vector<T> whole = GenWholeMatrixInput<T>(pattern, m, k, alongRow, lo, hi,
                                                  seed, wholeMatrix);
    if (wholeMatrix) {
        return whole;
    }
    return GenGroupBasedInput<T>(pattern, m, k, alongRow, violPos, lo, hi, rng);
}

}  // namespace sparse_test

#endif  // TEST_PRUNE_CHECK_GOLDEN_H_
