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
#ifndef SPSV_COMPLEX_SCALED_H_
#define SPSV_COMPLEX_SCALED_H_

#include "../common/spsv_complex_workspace.h"

#include "../../shared/spsv_workspace_layout.h"

#include "spsv_complex_scaled_tree.h"
#include "spsv_complex_certificate_schedule.h"

enum class SpsvScaledStatus { Unsupported, Pending, Certified };

struct SpsvScaledTree {
    int64_t leaves;
    int64_t count;
    int64_t words;
    int32_t width;
};

__aicore__ inline bool SpsvShouldTryScaledComplex(const SpsvTilingData& tiling, int32_t levels)
{
    return tiling.m >= 1024 && tiling.m <= 8192 && tiling.nnz / tiling.m <= 16 &&
           levels == tiling.m;
}

__aicore__ inline SpsvScaledTree SpsvSelectScaledTree(int64_t m, int32_t bandwidth)
{
    SpsvScaledTree best{0, 0, INT64_MAX, bandwidth};
    for (int32_t trial = bandwidth; trial <= kSpsvScaledMaxBandwidth; ++trial) {
        int64_t parts = (m + trial - 1) / trial;
        int64_t size = SpsvCertificateLeafCount(parts);
        int64_t words = ((size + parts) * trial * (trial + 1) + 2 * size * trial + m) * kSpsvScaledWords;
        if (words < best.words) {
            best = {size, parts, words, trial};
        }
    }
    return best;
}

template <int32_t WORDS, typename Data>
__aicore__ inline SpsvScaledStatus SpsvRunScaledTree(const Data& data, int64_t m, int64_t activeM,
                                                     const SpsvScaledTree& layout, int32_t block, int32_t blocks,
                                                     __gm__ uint32_t* risk)
{
    int32_t width = layout.width;
    int64_t leaves = layout.leaves, count = layout.count;
    __gm__ float* tree = (__gm__ float*)(risk + blocks);
    __gm__ float* state = tree + (leaves + count) * width * (width + 1) * kSpsvScaledWords;
    __gm__ float* inverse = state + 2 * leaves * width * kSpsvScaledWords;
    asc_vf_call<SpsvScaledSeed<WORDS, Data>>(dim3{128, 1, 1}, data, m, activeM, block, blocks, leaves, width, tree,
                                             inverse);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvScaledPrepare<WORDS, Data>>(dim3{256, 1, 1}, data, m, activeM, block, blocks, leaves, count, width,
                                                tree, inverse);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    // The root transform is unused: the root's incoming state is zero and
    // the downsweep reads only its left child's transform.
    for (int64_t level = leaves / 2; level >= 2; level /= 2) {
        asc_vf_call<SpsvScaledTreeUp<WORDS>>(dim3{256, 1, 1}, tree, leaves, count, level, width, block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    for (int64_t level = 1; level < leaves; level *= 2) {
        asc_vf_call<SpsvScaledTreeDown<WORDS>>(dim3{256, 1, 1}, tree, state, leaves, count, level, width, block,
                                               blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    asc_vf_call<SpsvScaledPublish<WORDS, Data>>(dim3{128, 1, 1}, data, m, activeM, tree, state, leaves, width, block,
                                                blocks, risk, inverse);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    if (activeM < m && block == 0) {
        asc_vf_call<SpsvScaledTerminalRow<WORDS, Data>>(dim3{256, 1, 1}, data, m, inverse, risk);
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return SpsvCollectCertificateFlags(risk, blocks) == 0U ? SpsvScaledStatus::Certified : SpsvScaledStatus::Pending;
}

__aicore__ inline bool SpsvPreferThreeScaledWords(const SpsvComplexCertificateData& data, uint32_t bandwidth)
{
    // Narrow affine transforms amortize a three-word proposal. Every omitted
    // term still enters its error enclosure; wider transforms retain four.
    return bandwidth <= (data.nonUnit ? 7U : 11U);
}

__aicore__ inline bool SpsvPreferThreeScaledWords(const SpsvRealScaledData&, uint32_t) { return false; }

__aicore__ inline bool SpsvScaledLayoutFits(int64_t m, int64_t activeM, int32_t blocks, const SpsvScaledTree& layout)
{
    uint64_t rows = static_cast<uint64_t>(m);
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(m)};
    uint64_t available = geometry.arenaWords;
    uint64_t used = geometry.riskWord + static_cast<uint64_t>(blocks) + static_cast<uint64_t>(layout.words) +
                    (m - activeM) * kSpsvScaledWords;
    return used <= available;
}

template <typename Data>
__aicore__ inline uint32_t SpsvScaledValidatedBandwidth(const Data& data, int64_t m, int32_t block, int32_t blocks,
                                                      __gm__ uint32_t* risk, bool ignoreTail)
{
    asc_vf_call<SpsvScaledValidate<Data>>(dim3{256, 1, 1}, data, m, block, blocks, risk, ignoreTail);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRefreshComplexCertificateFlags(risk);
    uint32_t bandwidth = 0;
    for (int32_t b = 0; b < blocks; ++b) {
        bandwidth = bandwidth > risk[b] ? bandwidth : risk[b];
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return bandwidth;
}

template <typename Data>
__aicore__ inline SpsvScaledStatus SpsvTryScaledComplex(const Data& data, int64_t m)
{
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    auto risk = SpsvComplexScratchView{(__gm__ float*)data.wide, {static_cast<uint64_t>(m)}}.risk;
    uint32_t bandwidth = SpsvScaledValidatedBandwidth(data, m, block, blocks, risk, false);
    if (bandwidth == 0 || bandwidth > kSpsvScaledMaxBandwidth + 1U) {
        return SpsvScaledStatus::Unsupported;
    }
    int64_t activeM = bandwidth == kSpsvScaledMaxBandwidth + 1U ? m - 1 : m;
    if (activeM < m) {
        bandwidth = SpsvScaledValidatedBandwidth(data, m, block, blocks, risk, true);
        if (bandwidth == 0) {
            bandwidth = 1;
        }
    }
    auto layout = SpsvSelectScaledTree(activeM, static_cast<int32_t>(bandwidth));
    if (!SpsvScaledLayoutFits(m, activeM, blocks, layout)) {
        return SpsvScaledStatus::Unsupported;
    }
    if (SpsvPreferThreeScaledWords(data, bandwidth)) {
        auto result = SpsvRunScaledTree<3>(data, m, activeM, layout, block, blocks, risk);
        if (result == SpsvScaledStatus::Certified) {
            return result;
        }
        // Retry with the existing four-word enclosure before the exact path.
        // No tolerance or overflow-sign certificate is weakened.
        return SpsvRunScaledTree<4>(data, m, activeM, layout, block, blocks, risk);
    }
    return SpsvRunScaledTree<4>(data, m, activeM, layout, block, blocks, risk);
}

#endif // SPSV_COMPLEX_SCALED_H_
