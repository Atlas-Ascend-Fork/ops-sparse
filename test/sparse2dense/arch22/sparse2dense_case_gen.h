// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE for the full text of the License.
// ============================================================================
// sparse2dense_case_gen.h — sparse2dense 用例数据构造（官方 sparse_patterns 十
// 模式权重与坐标展开/值生成/CSR-CSC-COO 三格式打包/CPU golden 存储计算）；
// 纯 host 计算，无设备/ACL 运行时调用；仅被 sparse2dense_test.cpp 使用。
// ============================================================================

#ifndef SPARSE2DENSE_CASE_GEN_H_
#define SPARSE2DENSE_CASE_GEN_H_

#include "sparse2dense_case_encode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// ============================================================================
// 坐标结构生成（官方 test_cases/common/sparse_patterns.py 十模式同构移植）
//   row_counts：按 pattern 权重把 nnz 分配到各行（每行 count <= cols，官方 row_counts
//   同款循环：share=floor(w/sum*remaining) 截断 + 全零时候选顺序补 1）；
//   段内列：uniform/highly_imbalanced/one_long_row/many_empty_rows/skewed/power_law →
//   start=(seed+row*17)%cols 步进 1；random → 官方 gcd 步进；banded → 以 (row%cols)
//   为中心连续段；block_like → 8 列块对齐；diagonal → 列=行号；段内升序（sorted_unique）。
// ============================================================================
inline std::vector<double> PatternWeights(RowPattern pattern, int64_t rows, int64_t cols, int64_t nnz, uint64_t seed)
{
    std::vector<double> w(static_cast<size_t>(rows), 1.0);
    switch (pattern) {
        case RowPattern::HIGHLY_IMBALANCED:
            if (rows > 0) {
                w[0] = 1000.0; // 官方 _weights：首行权重 1000
            }
            break;
        case RowPattern::ONE_LONG_ROW:
            if (rows > 0) {
                w[0] = 1.0e12; // 官方：首行权重 1e12
            }
            break;
        case RowPattern::MANY_EMPTY_ROWS: {
            const int64_t active = std::max<int64_t>(1, std::min(rows, (nnz + std::max<int64_t>(cols, 1) - 1) /
                std::max<int64_t>(cols, 1))); // 官方：ceil(nnz/cols) 行活跃
            for (int64_t r = 0; r < rows; ++r) {
                w[static_cast<size_t>(r)] = (r < active) ? 1.0 : 0.0;
            }
            break;
        }
        case RowPattern::SKEWED:
            for (int64_t r = 0; r < rows; ++r) {
                w[static_cast<size_t>(r)] = std::pow(2.0, static_cast<double>(std::min<int64_t>(r, 15)));
            }
            break;
        case RowPattern::POWER_LAW:
            for (int64_t r = 0; r < rows; ++r) {
                w[static_cast<size_t>(r)] = std::pow(static_cast<double>(r + 1), -1.25);
            }
            break;
        case RowPattern::RANDOM:
            for (int64_t r = 0; r < rows; ++r) { // 官方：hash 权重 + 1
                const int64_t hash = (r * 1103515245LL + static_cast<int64_t>(seed + 211ULL) * 12345LL) & 0x7fffffffLL;
                w[static_cast<size_t>(r)] = static_cast<double>(hash) + 1.0;
            }
            break;
        default: // uniform / diagonal / banded / block_like：全 1（官方 _weights 同口径）
            break;
    }
    return w;
}

// 单轮权重分配 pass（官方 row_counts 同款循环）：eligible 行（count < cols）按权重比例
// 分摊 share，全零分配时候选行顺序补 1；返回本轮结束后的已分配总数。remaining 为
// 本轮开始前剩余待分配数；无 eligible 权重时 ADD_FAILURE 并返回 -1 终止外层循环
inline int64_t AssignWeightsOnce(const CaseSpec &cs, const std::vector<double> &weights,
    std::vector<int64_t> &counts, int64_t remaining)
{
    std::vector<int64_t> share(counts.size(), 0);
    double sum = 0.0;
    for (int64_t r = 0; r < cs.rows; ++r) {
        if (counts[static_cast<size_t>(r)] < cs.cols) { // eligible（官方 active_weights 同口径）
            sum += weights[static_cast<size_t>(r)];
        }
    }
    if (sum == 0.0) {
        ADD_FAILURE() << "[case_id=" << cs.caseId << "] row pattern cannot represent requested nnz";
        return -1;
    }
    for (int64_t r = 0; r < cs.rows; ++r) {
        if (counts[static_cast<size_t>(r)] < cs.cols) {
            share[static_cast<size_t>(r)] = std::min(
                static_cast<int64_t>(std::floor(weights[static_cast<size_t>(r)] / sum *
                                                static_cast<double>(remaining))),
                cs.cols - counts[static_cast<size_t>(r)]);
        }
    }
    int64_t assigned = 0;
    for (int64_t r = 0; r < cs.rows; ++r) {
        assigned += share[static_cast<size_t>(r)];
    }
    if (assigned == 0) { // 官方同分支：eligible 候选中顺序补 1
        for (int64_t r = 0; r < cs.rows && assigned < remaining; ++r) {
            if (counts[static_cast<size_t>(r)] < cs.cols && weights[static_cast<size_t>(r)] > 0.0) {
                share[static_cast<size_t>(r)] = 1;
                assigned++;
            }
        }
    }
    int64_t total = 0;
    for (int64_t r = 0; r < cs.rows; ++r) {
        counts[static_cast<size_t>(r)] += share[static_cast<size_t>(r)];
        total += counts[static_cast<size_t>(r)];
    }
    return total;
}

inline std::vector<int64_t> RowCounts(const CaseSpec &cs)
{
    if (cs.nnz == 0) {
        return std::vector<int64_t>(static_cast<size_t>(cs.rows), 0);
    }
    if (cs.pattern == RowPattern::DIAGONAL) { // 官方：counts[r]=1（r<nnz），列=行号
        std::vector<int64_t> counts(static_cast<size_t>(cs.rows), 0);
        for (int64_t r = 0; r < std::min(cs.rows, cs.nnz); ++r) {
            counts[static_cast<size_t>(r)] = 1;
        }
        return counts;
    }
    const std::vector<double> weights = PatternWeights(cs.pattern, cs.rows, cs.cols, cs.nnz, cs.seed);
    std::vector<int64_t> counts(static_cast<size_t>(cs.rows), 0);
    int64_t remaining = cs.nnz;
    while (remaining > 0) {
        const int64_t total = AssignWeightsOnce(cs, weights, counts, remaining);
        if (total < 0) {
            return counts;  // AssignWeightsOnce 已 ADD_FAILURE（无 eligible 权重）
        }
        remaining = cs.nnz - total;
    }
    return counts;
}

inline int64_t GcdInt(int64_t a, int64_t b)
{
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

inline std::vector<int64_t> RowColumns(const CaseSpec &cs, int64_t row, int64_t count) // 段内升序列（0-based 逻辑列）
{
    std::vector<int64_t> columns;
    if (count <= 0) {
        return columns;
    }
    if (cs.pattern == RowPattern::DIAGONAL) {
        columns.push_back(row % cs.cols); // 官方：columns=[row]
    } else if (cs.pattern == RowPattern::BANDED) {
        const int64_t start = (row % cs.cols) - count / 2;
        for (int64_t k = 0; k < count; ++k) {
            columns.push_back(((k + start) % cs.cols + cs.cols) % cs.cols);
        }
    } else if (cs.pattern == RowPattern::BLOCK_LIKE) {
        const int64_t block = std::max<int64_t>(1, std::min<int64_t>(cs.cols, 8));
        const int64_t start = ((row / block) * block) % cs.cols;
        for (int64_t k = 0; k < count; ++k) {
            columns.push_back((k + start) % cs.cols);
        }
    } else if (cs.pattern == RowPattern::RANDOM) { // 官方：gcd 步进列生成
        const int64_t start = (static_cast<int64_t>(cs.seed) * 1103515245LL + row * 12345LL) % cs.cols;
        int64_t step = 1 + ((static_cast<int64_t>(cs.seed) + row) * 2654435761LL) % std::max<int64_t>(1, cs.cols - 1);
        while (GcdInt(step, cs.cols) != 1) {
            step += 1;
        }
        for (int64_t k = 0; k < count; ++k) {
            columns.push_back(((k * step + start) % cs.cols + cs.cols) % cs.cols);
        }
    } else { // uniform / highly_imbalanced / one_long_row / many_empty_rows / skewed / power_law
             // （官方 else 分支：start=(seed+row*17)%cols, step=1）
        const int64_t start = (static_cast<int64_t>(cs.seed) + row * 17) % cs.cols;
        for (int64_t k = 0; k < count; ++k) {
            columns.push_back((k + start) % cs.cols);
        }
    }
    std::sort(columns.begin(), columns.end()); // 段内唯一（count<=cols）且升序
    return columns;
}

// 逻辑坐标表（行主序展开；CSR/CSC/COO 共用同一坐标集合，官方 _sparse_inputs 同口径）
struct Coord {
    int64_t r;
    int64_t c;
};

inline std::vector<Coord> LogicalCoords(const CaseSpec &cs)
{
    std::vector<Coord> coords;
    const std::vector<int64_t> counts = RowCounts(cs);
    for (int64_t r = 0; r < cs.rows; ++r) {
        for (int64_t c : RowColumns(cs, r, counts[static_cast<size_t>(r)])) {
            coords.push_back({r, c});
        }
    }
    return coords;
}

// ============================================================================
// 稀疏存储构造（按 format；存储坐标 + base；CSC 列主序重排对齐官方 adapter）
// ============================================================================
struct SparseData {
    std::vector<int32_t> primary;   // CSR rowOffsets / CSC colOffsets / COO rowInd（存储坐标含 base）
    std::vector<int32_t> secondary; // CSR colInd / CSC rowInd / COO colInd
    std::vector<uint8_t> values;    // nnz × elemSize（原始 bit）
};

// 单元素 (re, im) 逻辑值生成（按 VKind 分布标签；values 流独立种子，与坐标乱序解耦）
inline void NextValue(const CaseSpec &cs, Rng &rng, int64_t i, double &re, double &im)
{
    const float qnan = std::numeric_limits<float>::quiet_NaN();  // 0x7FC00000（NaN 按原始 bit 搬运）
    const float pinf = std::numeric_limits<float>::infinity();   // +inf
    const float ninf = -std::numeric_limits<float>::infinity();  // -inf
    re = 0.0;
    im = 0.0;
    switch (cs.vkind) {
        case VKind::EMPTY: // nnz=0：无元素生成（GenValues 上方早退兜底）
            break;
        case VKind::UNIFORM: // uniform70：[vLo,vHi] 均匀；complex64 实/虚独立
            re = rng.NextUniform(cs.vLo, cs.vHi);
            im = rng.NextUniform(cs.vLo, cs.vHi);
            break;
        case VKind::SP_NAN: // 元素 0 = NaN（含 payload），其余 uniform
            re = (i == 0) ? static_cast<double>(qnan) : rng.NextUniform(cs.vLo, cs.vHi);
            im = (i == 0) ? rng.NextUniform(cs.vLo, cs.vHi) : rng.NextUniform(cs.vLo, cs.vHi);
            break;
        case VKind::SP_PINF:
            re = (i == 0) ? static_cast<double>(pinf) : rng.NextUniform(cs.vLo, cs.vHi);
            im = rng.NextUniform(cs.vLo, cs.vHi);
            break;
        case VKind::SP_NINF:
            re = (i == 0) ? static_cast<double>(ninf) : rng.NextUniform(cs.vLo, cs.vHi);
            im = rng.NextUniform(cs.vLo, cs.vHi);
            break;
        case VKind::ALL_NEG_ZERO: // 全 -0（浮点 4 dtype；-0 bit 与 +0 填充可区分）
            re = -0.0;
            im = -0.0;
            break;
        case VKind::ALL_POS_ZERO: // 全 +0（输出 ≡ 零 nnz 输出 bit）
            re = 0.0;
            im = 0.0;
            break;
        case VKind::C64_INDEP: { // L1_074：实/虚独立注入 +0/-0/NaN/±inf（确定性轮转）
            static const double special[5] = {0.0, -0.0, static_cast<double>(qnan),
                static_cast<double>(pinf), static_cast<double>(ninf)};
            re = special[(static_cast<size_t>(i) * 2U + 1U) % 5U];
            im = special[(static_cast<size_t>(i) * 3U + 4U) % 5U];
            break;
        }
    }
}

inline std::vector<uint8_t> GenValues(const CaseSpec &cs)
{
    const size_t esz = ElemBytes(cs.dtype);
    std::vector<uint8_t> v(static_cast<size_t>(cs.nnz) * esz, 0);
    if (cs.vkind == VKind::EMPTY || cs.nnz == 0) {
        return v;
    }
    Rng rng(cs.seed); // values 独立种子流（与坐标乱序解耦；官方 _values 同口径单分布）
    for (int64_t i = 0; i < cs.nnz; ++i) {
        double re = 0.0;
        double im = 0.0;
        NextValue(cs, rng, i, re, im);
        QuantizeStore(cs, re, im, v.data() + static_cast<size_t>(i) * esz);
    }
    return v;
}

// ---- CSR：rowOffsets 端点累加 + colInd 行主序；显式 offsets 用例直接由 offsets 展开 ----
inline SparseData BuildCsr(const CaseSpec &cs, const std::vector<Coord> &logical)
{
    SparseData sd;
    const int32_t base = static_cast<int32_t>(cs.base);
    sd.values = GenValues(cs);
    if (cs.explicitPrimary != nullptr) { // L0_009/L1_063：显式 rowOffsets（含空行）——坐标直接由 offsets 展开
        sd.primary.assign(cs.explicitPrimary, cs.explicitPrimary + cs.explicitPrimaryLen);
        for (int64_t r = 0; r < cs.rows; ++r) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(r)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(r) + 1]) - base;
            for (int64_t c : RowColumns(cs, r, e - s)) {
                sd.secondary.push_back(static_cast<int32_t>(c + base));
            }
        }
        return sd;
    }
    sd.primary.assign(static_cast<size_t>(cs.rows) + 1, 0);
    for (size_t k = 0; k < logical.size(); ++k) { // logical 按行主序 → offsets 端点累加
        sd.primary[static_cast<size_t>(logical[k].r) + 1]++;
    }
    for (int64_t r = 0; r < cs.rows; ++r) {
        sd.primary[static_cast<size_t>(r) + 1] += sd.primary[static_cast<size_t>(r)];
    }
    for (size_t k = 0; k < sd.primary.size(); ++k) { // 端点加 base（存储坐标含 base）
        sd.primary[k] += base;
    }
    sd.secondary.reserve(logical.size());
    for (const Coord &p : logical) {
        sd.secondary.push_back(static_cast<int32_t>(p.c + base));
    }
    return sd;
}

// ---- CSC：列主序重排对齐官方 adapter（values = values[order]）；显式 offsets 用例
//      直接由 colOffsets 展开（段内行号 (seed+c*17)%rows 步进 1，段内升序唯一）----
inline SparseData BuildCsc(const CaseSpec &cs, const std::vector<Coord> &logical)
{
    SparseData sd;
    const int32_t base = static_cast<int32_t>(cs.base);
    sd.values = GenValues(cs);
    if (cs.explicitPrimary != nullptr) { // L0_010/L1_064：显式 colOffsets（含空列）——坐标直接由 offsets 展开
        sd.primary.assign(cs.explicitPrimary, cs.explicitPrimary + cs.explicitPrimaryLen);
        for (int64_t c = 0; c < cs.cols; ++c) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(c)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(c) + 1]) - base;
            std::vector<int64_t> rows;
            const int64_t start = (static_cast<int64_t>(cs.seed) + c * 17) % cs.rows;
            for (int64_t k = 0; k < e - s; ++k) {
                rows.push_back((k + start) % cs.rows);
            }
            std::sort(rows.begin(), rows.end()); // 段内升序（sorted_unique；count<=rows 保证唯一）
            for (int64_t r : rows) {
                sd.secondary.push_back(static_cast<int32_t>(r + base));
            }
        }
        return sd;
    }
    std::vector<size_t> order(logical.size());
    for (size_t k = 0; k < order.size(); ++k) {
        order[k] = k;
    }
    std::stable_sort(order.begin(), order.end(), [&logical, &cs](size_t a, size_t b) {
        return (logical[a].c * cs.rows + logical[a].r) < (logical[b].c * cs.rows + logical[b].r);
    });
    sd.primary.assign(static_cast<size_t>(cs.cols) + 1, 0);
    for (size_t k = 0; k < order.size(); ++k) { // 列主序 counts 端点
        sd.primary[static_cast<size_t>(logical[order[k]].c) + 1]++;
    }
    for (int64_t c = 0; c < cs.cols; ++c) {
        sd.primary[static_cast<size_t>(c) + 1] += sd.primary[static_cast<size_t>(c)];
    }
    for (size_t k = 0; k < sd.primary.size(); ++k) { // 端点加 base（存储坐标含 base）
        sd.primary[k] += base;
    }
    sd.secondary.reserve(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
        sd.secondary.push_back(static_cast<int32_t>(logical[order[k]].r + base));
    }
    // values 与坐标同步重排（官方 adapter：values = values[order]）
    const size_t esz = ElemBytes(cs.dtype);
    std::vector<uint8_t> reordered(sd.values.size(), 0);
    for (size_t k = 0; k < order.size(); ++k) {
        CopyBytes(reordered.data() + k * esz, sd.values.data() + static_cast<size_t>(order[k]) * esz, esz);
    }
    sd.values.swap(reordered);
    return sd;
}

// ---- COO：primary=rowInd / secondary=colInd；shuffled_unique 用例按 Fisher-Yates 乱序、
//      reversed_sorted_unique 用例逆序（唯一性保持，坐标-值配对随序整体反转）----
inline SparseData BuildCoo(const CaseSpec &cs, const std::vector<Coord> &logical)
{
    SparseData sd;
    const int32_t base = static_cast<int32_t>(cs.base);
    sd.values = GenValues(cs);
    std::vector<size_t> order(logical.size());
    for (size_t k = 0; k < order.size(); ++k) {
        order[k] = k;
    }
    if (cs.coords == CoordMode::SHUFFLED_UNIQUE) {
        Rng rng(cs.seed + 1); // 坐标乱序独立流（values 流不受影响）
        for (size_t a = order.size(); a > 1; --a) {
            const size_t b = static_cast<size_t>(rng.NextInt(0, static_cast<int64_t>(a) - 1));
            std::swap(order[a - 1], order[b]);
        }
    } else if (cs.coords == CoordMode::REVERSED_UNIQUE) {
        for (size_t k = 0; k < order.size(); ++k) { // 同坐标集合逆序（L1_065 等价断言）
            order[k] = order.size() - 1U - k;
        }
    }
    sd.primary.reserve(order.size());
    sd.secondary.reserve(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
        sd.primary.push_back(static_cast<int32_t>(logical[order[k]].r + base));
        sd.secondary.push_back(static_cast<int32_t>(logical[order[k]].c + base));
    }
    if (cs.coords != CoordMode::SORTED_UNIQUE) { // values 与坐标同步重排（坐标-值配对保持；官方 adapter 同口径）
        const size_t esz = ElemBytes(cs.dtype);
        std::vector<uint8_t> reordered(sd.values.size(), 0);
        for (size_t k = 0; k < order.size(); ++k) {
            CopyBytes(reordered.data() + k * esz, sd.values.data() + static_cast<size_t>(order[k]) * esz, esz);
        }
        sd.values.swap(reordered);
    }
    return sd;
}

inline SparseData BuildSparse(const CaseSpec &cs, const std::vector<Coord> &logical)
{
    if (cs.fmt == Fmt::CSR) {
        return BuildCsr(cs, logical);
    }
    if (cs.fmt == Fmt::CSC) {
        return BuildCsc(cs, logical);
    }
    return BuildCoo(cs, logical);
}

// ============================================================================
// CPU Golden（与被测同 dtype 的 canonical 转换 → 按布局写入期望存储）
//   golden 逻辑矩阵 zeros((rows,cols)) 后按 format 展开 scatter；期望存储按 DnMat 布局寻址
//   （ROW dense[r*ld+c] / COL dense[c*ld+r]），ld padding 区恒 +0（清零 kernel 语义）。
// ============================================================================
inline std::vector<uint8_t> ComputeGoldenStorage(const CaseSpec &cs, const SparseData &sd)
{
    const size_t esz = ElemBytes(cs.dtype);
    const size_t logicalBytes = static_cast<size_t>(cs.rows * cs.cols) * esz;
    std::vector<uint8_t> dense(logicalBytes, 0); // +0 填充（dtype 正零 bit 模式）
    const int64_t base = cs.base;
    if (cs.fmt == Fmt::CSR) {
        for (int64_t r = 0; r < cs.rows; ++r) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(r)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(r) + 1]) - base;
            for (int64_t k = s; k < e; ++k) {
                const int64_t c = static_cast<int64_t>(sd.secondary[static_cast<size_t>(k)]) - base;
                CopyBytes(dense.data() + static_cast<size_t>(r * cs.cols + c) * esz,
                          sd.values.data() + static_cast<size_t>(k) * esz, esz);
            }
        }
    } else if (cs.fmt == Fmt::CSC) {
        for (int64_t c = 0; c < cs.cols; ++c) {
            const int64_t s = static_cast<int64_t>(sd.primary[static_cast<size_t>(c)]) - base;
            const int64_t e = static_cast<int64_t>(sd.primary[static_cast<size_t>(c) + 1]) - base;
            for (int64_t k = s; k < e; ++k) {
                const int64_t r = static_cast<int64_t>(sd.secondary[static_cast<size_t>(k)]) - base;
                CopyBytes(dense.data() + static_cast<size_t>(r * cs.cols + c) * esz,
                          sd.values.data() + static_cast<size_t>(k) * esz, esz);
            }
        }
    } else {
        for (size_t k = 0; k < sd.primary.size(); ++k) {
            const int64_t r = static_cast<int64_t>(sd.primary[k]) - base;
            const int64_t c = static_cast<int64_t>(sd.secondary[k]) - base;
            CopyBytes(dense.data() + static_cast<size_t>(r * cs.cols + c) * esz,
                       sd.values.data() + k * esz, esz);
        }
    }
    // 逻辑矩阵 → 期望存储（含 padding；全 +0）
    const int64_t ld = cs.ld;
    const size_t storageElems =
        static_cast<size_t>((cs.layout == Layout::ROW) ? (cs.rows * ld) : (cs.cols * ld));
    std::vector<uint8_t> storage(storageElems * esz, 0);
    for (int64_t r = 0; r < cs.rows; ++r) {
        for (int64_t c = 0; c < cs.cols; ++c) {
            const int64_t off = (cs.layout == Layout::ROW) ? (r * ld + c) : (c * ld + r);
            CopyBytes(storage.data() + static_cast<size_t>(off) * esz,
                       dense.data() + static_cast<size_t>(r * cs.cols + c) * esz, esz);
        }
    }
    return storage;
}

#endif // SPARSE2DENSE_CASE_GEN_H_
