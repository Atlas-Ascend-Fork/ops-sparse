/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "internal/csr_conversion.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <limits>
#define CHECK(x)                                           \
    do {                                                   \
        if (!(x))                                          \
            throw std::runtime_error("CHECK failed: " #x); \
    } while (0)
template <class F>
void rejects(F f)
{
    bool ok = false;
    try {
        f();
    } catch (const std::exception&) {
        ok = true;
    }
    CHECK(ok);
}
using namespace ascspmm;
void CheckCanonicalCsr(const Csr& a, std::vector<std::map<int, double>>& ref)
{
    auto v = convert(a);
    std::vector<float> dense(v.mpad * a.k, 0);
    int64_t count = 0, pmax = 0;
    for (int b = 0; b < v.mpad / 16; ++b) {
        pmax = std::max<int64_t>(pmax, v.rowptr[b + 1] - v.rowptr[b]);
        int last = -1;
        for (int p = v.rowptr[b]; p < v.rowptr[b + 1]; ++p) {
            CHECK(v.colidx[p] > last);
            last = v.colidx[p];
            bool active = false;
            for (int r = 0; r < 16; ++r) {
                auto f = from_half(v.values[p * 16 + r]);
                dense[(b * 16 + r) * a.k + last] = f;
                count += f != 0;
                active |= f != 0;
            }
            CHECK(active);
        }
    }
    for (int r = 0; r < v.mpad; ++r) {
        for (int c = 0; c < a.k; ++c) {
            CHECK(dense[r * a.k + c] == (r < a.m ? from_half(to_half(ref[r][c])) : 0));
        }
    }
    CHECK(count == v.stats.fp16_nnz);
    CHECK(pmax == v.pmax);
}

TEST(CubeSpmm1dConversion, FiniteHalfAndRounding)
{
    for (unsigned i = 0; i < 65536; ++i) {
        if ((i & 0x7c00) != 0x7c00) {
            CHECK(to_half(from_half(i)) == i);
        }
    }
    CHECK(to_half(1. + std::ldexp(1., -11)) == 0x3c00);
    CHECK(to_half(1. + 3 * std::ldexp(1., -11)) == 0x3c02);
    CHECK(to_half(std::ldexp(1., -25)) == 0);
    CHECK(to_half(3 * std::ldexp(1., -25)) == 2);
    rejects([] { to_half(65505); });
    rejects([] { to_half(INFINITY); });
    rejects([] { to_half(NAN); });
}

TEST(CubeSpmm1dConversion, RandomCsr)
{
    std::mt19937 gen(592);
    for (int trial = 0; trial < 1000; ++trial) {
        Csr a;
        a.m = 1 + gen() % 65;
        a.k = 1 + gen() % 81;
        a.rowptr.push_back(0);
        std::vector<std::map<int, double>> ref(a.m);
        for (int r = 0; r < a.m; ++r) {
            int len = (trial % 9 == 0 || (r >= 16 && r < 32)) ? 0 : gen() % 50;
            for (int z = 0; z < len; ++z) {
                int c = gen() % a.k;
                double v = (int(gen() % 33) - 16) / 8.;
                a.colidx.push_back(c);
                a.values.push_back(v);
                ref[r][c] += v;
            }
            a.rowptr.push_back(a.colidx.size());
        }
        CheckCanonicalCsr(a, ref);
    }
}

TEST(CubeSpmm1dConversion, InvalidCsrAndEmptyCases)
{
    Csr a{17, 3, {0}, {}, {}};
    a.rowptr.resize(18, 0);
    auto empty = convert(a);
    CHECK(empty.mpad == 32);
    CHECK(empty.rowptr.size() == 3);
    CHECK(empty.pmax == 0);
    rejects([&] {
        auto b = a;
        b.rowptr[2] = -1;
        convert(b);
    });
    rejects([&] {
        auto b = a;
        b.rowptr.back() = 1;
        convert(b);
    });
    rejects([&] {
        auto b = a;
        b.rowptr[0] = 1;
        convert(b);
    });
    rejects([&] {
        auto b = a;
        b.m = INT32_MAX;
        convert(b);
    });
    rejects([&] {
        auto b = a;
        b.k = 0;
        convert(b);
    });
    Csr one{1, 1, {0, 1}, {1}, {1}};
    rejects([&] { convert(one); });
    one.colidx[0] = 0;
    one.values[0] = NAN;
    rejects([&] { convert(one); });
    one.values[0] = 1e-20;
    auto sub = convert(one);
    CHECK(sub.stats.underflow == 1 && sub.colidx.empty());
    Csr cancel{1, 1, {0, 3}, {0, 0, 0}, {4, -4, 0}};
    CHECK(convert(cancel).colidx.empty());
    rejects([] { checked_bytes(UINT64_MAX, 32); });
}
