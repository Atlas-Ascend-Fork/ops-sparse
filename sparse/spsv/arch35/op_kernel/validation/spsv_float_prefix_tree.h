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

#ifndef SPSV_FLOAT_PREFIX_TREE_H_
#define SPSV_FLOAT_PREFIX_TREE_H_

#include "../common/spsv_simt_common.h"

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline int64_t SpsvRealPrefixAddress(int64_t block, int32_t row, int32_t column)
{
    return (block * (WIDTH * (WIDTH + 1)) + row * (WIDTH + 1) + column) * 4;
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvRealTreeLoad(__gm__ const float* tree, int64_t node, int32_t row,
                                                              int32_t column, int64_t leaves, int64_t count)
{
    if (node >= leaves + count) {
        SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
        if (row == column) {
            z.re.hi = 1.f;
        }
        return z;
    }
    return SpsvRealLoad(tree + SpsvRealPrefixAddress<WIDTH>(node, row, column));
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__simt_vf__ __aicore__ __launch_bounds__(512) inline void SpsvRealTreeUp(__gm__ float* tree, int64_t leaves,
                                                                         int64_t count, int64_t first, int32_t block,
                                                                         int32_t blocks)
{
    SpsvSimtRefresh(tree);
    for (int64_t entry = static_cast<int64_t>(block) * 512 + threadIdx.x; entry < first * (WIDTH * (WIDTH + 1));
         entry += static_cast<int64_t>(blocks) * 512) {
        auto position = SpsvAffineUpPosition<WIDTH>(entry, first);
        SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
        if (position.column == WIDTH) {
            z = SpsvRealTreeLoad<WIDTH>(tree, 2 * position.node + 1, position.row, position.column, leaves, count);
        }
        for (int32_t k = 0; k < WIDTH; ++k) {
            z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixTreeMul<SHORT_PRODUCT>(
                                           SpsvRealTreeLoad<WIDTH>(tree, 2 * position.node + 1, position.row, k, leaves, count),
                                           SpsvRealTreeLoad<WIDTH>(tree, 2 * position.node, k, position.column, leaves, count)));
        }
        SpsvRealStore(tree + SpsvRealPrefixAddress<WIDTH>(position.node, position.row, position.column), z);
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvRealTreeState(__gm__ const float* states, int64_t node, int32_t row)
{
    if (node == 1) {
        return SpsvTriplePrefixZero();
    }
    return SpsvRealLoad(states + (node * WIDTH + row) * 4);
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvRealTreeDown(__gm__ const float* tree,
                                                                           __gm__ float* states, int64_t leaves,
                                                                           int64_t count, int64_t first, int32_t block,
                                                                           int32_t blocks)
{
    SpsvSimtRefresh(tree);
    for (int64_t entry = static_cast<int64_t>(block) * 256 + threadIdx.x; entry < 2 * first * WIDTH;
         entry += static_cast<int64_t>(blocks) * 256) {
        int64_t child = 2 * first + entry / WIDTH;
        int64_t parent = child / 2;
        int32_t row = static_cast<int32_t>(entry % WIDTH);
        SpsvTriplePrefixValue z = SpsvRealTreeState<WIDTH>(states, parent, row);
        if ((child & 1) != 0) {
            z = SpsvRealTreeLoad<WIDTH>(tree, child - 1, row, WIDTH, leaves, count);
            for (int32_t k = 0; k < WIDTH; ++k) {
                z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixTreeMul<SHORT_PRODUCT>(
                                               SpsvRealTreeLoad<WIDTH>(tree, child - 1, row, k, leaves, count),
                                               SpsvRealTreeState<WIDTH>(states, parent, k)));
            }
        }
        SpsvRealStore(states + (child * WIDTH + row) * 4, z);
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvRealTreeResult(__gm__ const float* tree, __gm__ const float* states,
                                                                int64_t leaves, int64_t at)
{
    int64_t node = leaves + at / WIDTH;
    int32_t row = static_cast<int32_t>(at % WIDTH);
    SpsvTriplePrefixValue z = SpsvRealLoad(tree + SpsvRealPrefixAddress<WIDTH>(node, row, WIDTH));
    for (int32_t k = 0; k < WIDTH; ++k) {
        z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixMul(SpsvRealLoad(tree + SpsvRealPrefixAddress<WIDTH>(node, row, k)),
                                                       SpsvRealTreeState<WIDTH>(states, node, k)));
    }
    return z;
}

#endif // SPSV_FLOAT_PREFIX_TREE_H_
