/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_conversion.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace ascspmm {
size_t checked_bytes(uint64_t count, size_t element)
{
    if (element != 0 && count > std::numeric_limits<size_t>::max() / element) {
        throw std::overflow_error("byte size overflow");
    }
    return static_cast<size_t>(count) * element;
}
static uint32_t round_even(double x)
{
    auto lo = static_cast<uint32_t>(std::floor(x));
    const double frac = x - lo;
    return lo + (frac > .5 || (frac == .5 && (lo & 1)));
}
uint16_t to_half(double x)
{
    if (!std::isfinite(x) || std::abs(x) > 65504.0) {
        throw std::invalid_argument("nonfinite or FP16 range overflow");
    }
    const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    x = std::abs(x);
    if (x < std::ldexp(1., -14)) {
        return sign | static_cast<uint16_t>(round_even(std::ldexp(x, 24)));
    }
    int e;
    std::frexp(x, &e);
    --e;
    auto mant = round_even(std::ldexp(x, 10 - e));
    if (mant == 2048) {
        mant = 1024;
        ++e;
    }
    return sign | static_cast<uint16_t>(((e + 15) << 10) | (mant - 1024));
}
float from_half(uint16_t b)
{
    int e = (b >> 10) & 31, v = b & 1023;
    float x = e == 0 ?
                  std::ldexp(static_cast<float>(v), -24) :
                  (e == 31 ? (v ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
                             std::ldexp(float(v + 1024), e - 25));
    return (b & 0x8000) ? -x : x;
}

namespace {
void ValidateCsr(const Csr& a)
{
    constexpr int64_t max32 = std::numeric_limits<int32_t>::max();
    if (a.m <= 0 || a.m > max32 - 15 || a.k <= 0 || a.k > max32) {
        throw std::invalid_argument("positive M/K and int32 padded dimensions required");
    }
    if (a.rowptr.size() != static_cast<size_t>(a.m) + 1 || a.rowptr[0] != 0 || a.colidx.size() != a.values.size() ||
        a.values.size() > uint64_t(INT64_MAX) || a.rowptr.back() != static_cast<int64_t>(a.values.size())) {
        throw std::invalid_argument("CSR lengths/endpoints mismatch");
    }
    for (size_t r = 1; r < a.rowptr.size(); ++r) {
        if (a.rowptr[r] < a.rowptr[r - 1] || a.rowptr[r] > a.rowptr.back()) {
            throw std::invalid_argument("nonmonotone CSR rowptr");
        }
    }
}

struct CanonicalRows {
    std::vector<int64_t> rp;
    std::vector<int32_t> ci;
    std::vector<uint16_t> va;
    std::vector<std::pair<int32_t, double>> row;

    void AppendRow(Stats& stats)
    {
        auto byColumn = [](const auto& x, const auto& y) { return x.first < y.first; };
        if (!std::is_sorted(row.begin(), row.end(), byColumn)) {
            std::stable_sort(row.begin(), row.end(), byColumn);
        }
        for (size_t i = 0; i < row.size();) {
            const auto col = row[i].first;
            double sum = 0;
            do {
                sum += row[i++].second;
                if (!std::isfinite(sum)) {
                    throw std::invalid_argument("duplicate sum overflow");
                }
            } while (i < row.size() && row[i].first == col);
            if (sum == 0) {
                continue;
            }
            ++stats.canonical_nnz;
            const uint16_t bits = to_half(sum);
            if ((bits & 0x7fff) == 0) {
                ++stats.underflow;
                continue;
            }
            ci.push_back(col);
            va.push_back(bits);
        }
    }

    CanonicalRows(const Csr& a, Stats& stats) : rp(a.m + 1, 0)
    {
        for (int64_t r = 0; r < a.m; ++r) {
            row.clear();
            for (int64_t i = a.rowptr[r]; i < a.rowptr[r + 1]; ++i) {
                if (a.colidx[i] < 0 || a.colidx[i] >= a.k || !std::isfinite(a.values[i])) {
                    throw std::invalid_argument("invalid column or nonfinite input");
                }
                stats.explicit_zeros += (a.values[i] == 0);
                row.emplace_back(static_cast<int32_t>(a.colidx[i]), a.values[i]);
            }
            AppendRow(stats);
            rp[r + 1] = static_cast<int64_t>(ci.size());
        }
        stats.fp16_nnz = static_cast<int64_t>(ci.size());
    }
};

class BlockMerger {
public:
    BlockMerger(const CanonicalRows& rows, VectorCsr& out) : rows_(rows), out_(out) {}
    int64_t Merge(int64_t block, bool write)
    {
        return SharedPattern(block) ? MergeShared(block, write) : MergeGeneral(block, write);
    }

private:
    const CanonicalRows& rows_;
    VectorCsr& out_;
    using Entry = std::tuple<int32_t, int32_t, int64_t>;
    bool SharedPattern(int64_t block) const
    {
        const auto& rp = rows_.rp;
        const auto& ci = rows_.ci;
        const int64_t first = block * 16, length = rp[first + 1] - rp[first];
        for (int64_t r = first + 1; r < std::min(first + 16, int64_t(out_.m)); ++r) {
            if (rp[r + 1] - rp[r] != length ||
                !std::equal(ci.begin() + rp[first], ci.begin() + rp[first + 1], ci.begin() + rp[r])) {
                return false;
            }
        }
        return true;
    }
    int64_t MergeShared(int64_t block, bool write)
    {
        const auto& rp = rows_.rp;
        const int64_t first = block * 16, count = rp[first + 1] - rp[first];
        if (write) {
            for (int64_t j = 0; j < count; ++j) {
                const int64_t target = static_cast<int64_t>(out_.rowptr[block]) + j;
                out_.colidx[target] = rows_.ci[rp[first] + j];
                for (int64_t r = first; r < std::min(first + 16, int64_t(out_.m)); ++r) {
                    out_.values[target * 16 + r - first] = rows_.va[rp[r] + j];
                }
            }
        }
        return count;
    }
    int64_t MergeGeneral(int64_t block, bool write)
    {
        const auto& rp = rows_.rp;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
        for (int32_t r = 0; r < 16 && block * 16 + r < out_.m; ++r) {
            const auto pos = rp[block * 16 + r];
            if (pos < rp[block * 16 + r + 1]) {
                heap.emplace(rows_.ci[pos], r, pos);
            }
        }
        int64_t count = 0;
        while (!heap.empty()) {
            const int32_t col = std::get<0>(heap.top());
            const int64_t target = static_cast<int64_t>(out_.rowptr[block]) + count;
            if (write) {
                out_.colidx[target] = col;
            }
            do {
                auto [column, row, pos] = heap.top();
                heap.pop();
                (void)column;
                if (write) {
                    out_.values[target * 16 + row] = rows_.va[pos];
                }
                ++pos;
                if (pos < rp[block * 16 + row + 1]) {
                    heap.emplace(rows_.ci[pos], row, pos);
                }
            } while (!heap.empty() && std::get<0>(heap.top()) == col);
            ++count;
        }
        return count;
    }
};
} // namespace

VectorCsr convert(const Csr& a)
{
    ValidateCsr(a);
    VectorCsr out;
    out.m = static_cast<int32_t>(a.m);
    out.k = static_cast<int32_t>(a.k);
    out.mpad = static_cast<int32_t>((a.m + 15) / 16 * 16);
    out.stats.stored = static_cast<int64_t>(a.values.size());
    const CanonicalRows canonical(a, out.stats);
    const int64_t blocks = out.mpad / 16;
    out.rowptr.resize(blocks + 1, 0);
    BlockMerger merger(canonical, out);
    for (int64_t b = 0; b < blocks; ++b) {
        const int64_t count = merger.Merge(b, false);
        if (count + out.rowptr[b] > INT32_MAX / 16) {
            throw std::overflow_error("Vector-CSR value offset exceeds int32");
        }
        out.rowptr[b + 1] = static_cast<int32_t>(count + out.rowptr[b]);
        out.pmax = std::max(out.pmax, count);
        out.stats.empty_blocks += count == 0;
    }
    const uint64_t vectors = out.rowptr.back();
    checked_bytes(vectors, 16 * sizeof(uint16_t));
    out.colidx.resize(vectors);
    out.values.resize(vectors * 16, 0);
    for (int64_t b = 0; b < blocks; ++b) {
        (void)merger.Merge(b, true);
    }
    return out;
}
} // namespace ascspmm
