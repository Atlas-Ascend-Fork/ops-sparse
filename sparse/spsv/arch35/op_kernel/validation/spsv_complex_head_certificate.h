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

#ifndef SPSV_COMPLEX_HEAD_CERTIFICATE_H_
#define SPSV_COMPLEX_HEAD_CERTIFICATE_H_

#include "../common/spsv_complex_workspace.h"

#include "spsv_complex_border_schedule.h"

// A bounded arbitrary head precedes a banded body. Full original-equation
// residuals and outward-rounded radii remain in the signed recurrence.
template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvHeadClassify(const SpsvComplexCertificateData data,
                                                                           int64_t m, int32_t block, int32_t blocks,
                                                                           __gm__ uint32_t* flags)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.wide);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    uint32_t head = 0;
    for (int64_t at = static_cast<int64_t>(block) * 8 + warp; at < m; at += static_cast<int64_t>(blocks) * 8) {
        int64_t r = data.forward ? at : m - 1 - at;
        bool bad = !SpsvCertificateRowFinite(data, r, 65536);
        if (data.rowPtr[r + 1] - data.rowPtr[r] > 64) {
            head = head > at + 1 ? head : static_cast<uint32_t>(at + 1);
        }
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
            if (source < at - WIDTH) {
                head = head > source + 1 ? head : static_cast<uint32_t>(source + 1);
            }
            bad |= (p > data.rowPtr[r] && data.colInd[p] <= data.colInd[p - 1]);
            bad |= !SpsvBlockFiniteNormal(data.values[p].real) || !SpsvBlockFiniteNormal(data.values[p].imag);
        }
        if (bad) {
            head = 0x80000000U;
        }
    }
    head = asc_reduce_max(head);
    __ubuf__ uint32_t warpHeads[8];
    if (lane == 0) {
        warpHeads[warp] = head;
    }
    asc_syncthreads();
    if (threadIdx.x == 0) {
        for (int32_t w = 1; w < 8; ++w) {
            head = head > warpHeads[w] ? head : warpHeads[w];
        }
        flags[block] = head;
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_callee__ inline SpsvTriplePrefixValue SpsvHeadBodyRow(const SpsvComplexCertificateData& data, int64_t m,
                                                             int64_t head, int64_t part, int32_t row, int32_t tid,
                                                             __gm__ float* buffer, __gm__ const float* inverses,
                                                             __gm__ const float* defects,
                                                             __gm__ const float* corrections)
{
    int64_t logical = head + part * WIDTH + row;
    int64_t r = data.forward ? logical : m - 1 - logical;
    SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
    if (logical >= m) {
        return z;
    }
    if (tid == WIDTH) {
        z = SpsvTriplePrefixLoad(defects + logical * 7);
    }
    SpsvTriplePrefixValue dependencies = SpsvTriplePrefixZero();
    for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
        int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
        if (source >= logical) {
            continue;
        }
        SpsvTriplePrefixValue value = SpsvTriplePrefixZero();
        if (source < head) {
            if (tid == WIDTH) {
                value = SpsvTriplePrefixLoad(corrections + source * 7);
            }
        } else if (source >= head + part * WIDTH) {
            value = SpsvTriplePrefixLoad(
                buffer + SpsvTriplePrefixAddress<WIDTH>(part, static_cast<int32_t>(source - head - part * WIDTH), tid));
        } else if (source - head - (part - 1) * WIDTH == tid) {
            value.re.hi = 1.f;
        } else if (source < head + (part - 1) * WIDTH) {
            z.radius = __builtin_inff();
        }
        dependencies = SpsvTriplePrefixAdd(
            dependencies, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag, value));
    }
    if (data.nonUnit) {
        dependencies = SpsvTriplePrefixMul(dependencies, SpsvTriplePrefixLoad(inverses + logical * 7));
    }
    return SpsvTriplePrefixAdd(z, dependencies);
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvHeadBodyPrepare(
    const SpsvComplexCertificateData data, int64_t m, int64_t head, int32_t block, int32_t blocks, __gm__ float* buffer,
    __gm__ const float* inverses, __gm__ const float* defects, __gm__ const float* corrections)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)corrections);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int64_t count = (m - head + WIDTH - 1) / WIDTH;
    for (int64_t first = static_cast<int64_t>(block) * 8; first < count; first += static_cast<int64_t>(blocks) * 8) {
        int64_t part = first + warp;
        asc_syncthreads();
        if (part >= count || lane >= (WIDTH + 1)) {
            continue;
        }
        for (int32_t row = 0; row < WIDTH; ++row) {
            auto z = SpsvHeadBodyRow<WIDTH>(data, m, head, part, row, lane, buffer, inverses, defects, corrections);
            SpsvTriplePrefixStore(buffer + SpsvTriplePrefixAddress<WIDTH>(part, row, lane), z);
        }
    }
    asc_threadfence();
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvHeadBodyResults(__gm__ const float* tree,
                                                                              __gm__ const float* states,
                                                                              int64_t leaves, int64_t head, int64_t m,
                                                                              int32_t block, int32_t blocks,
                                                                              __gm__ float* corrections)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)tree);
    }
    asc_syncthreads();
    for (int64_t at = head + static_cast<int64_t>(block) * 128 + threadIdx.x; at < m;
         at += static_cast<int64_t>(blocks) * 128) {
        SpsvTriplePrefixStore(corrections + at * 7, SpsvTripleTreeResult<WIDTH>(tree, states, leaves, at - head));
    }
    asc_threadfence();
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvSelectHeadCertificate(const SpsvComplexCertificateData& data, int64_t m, uint32_t riskBlocks,
                                                 int32_t block, int32_t blocks, SpsvBorderCertificateState& state)
{
    if (!SpsvBeginBorderSelection(data, m, riskBlocks, blocks, state)) {
        return false;
    }
    uint32_t slots = state.slots;
    __gm__ uint32_t* flags = state.flags;
    asc_vf_call<SpsvHeadClassify<WIDTH>>(dim3{256, 1, 1}, data, m, block, blocks, flags);
    uint32_t edge = SpsvClassifiedCertificateEdge(flags, blocks);
    if (edge & 0x80000000U) {
        return false;
    }
    int64_t head = (static_cast<int64_t>(edge) + 31) / 32 * 32;
    if (head < 32 || head > 512 || head >= m) {
        return false;
    }
    state.head = head;
    state.slots = slots;
    state.flags = flags;
    return true;
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvAllocateHeadCertificate(int64_t m, int32_t blocks, SpsvBorderCertificateState& state)
{
    int64_t head = state.head;
    auto arena = SpsvMakeBorderArena<WIDTH>(m, (m - head + WIDTH - 1) / WIDTH, m, blocks, state);
    arena.local = arena.constants + 7 * head;
    uint64_t required = SpsvComplexWorkspaceGeometry{static_cast<uint64_t>(m)}.riskWord + state.slots + blocks +
                        2 * arena.leaves * (WIDTH * (WIDTH + 2) * kSpsvTriplePrefixWords) + 3 * 7 * m + 7 * head +
                        head * 32 * 7;
    if (!SpsvCertificateArenaFits(m, required)) {
        return false;
    }
    state = arena;
    return true;
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__aicore__ inline uint32_t SpsvRunHeadCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                  uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvBorderCertificateState state{};
    if (!SpsvSelectHeadCertificate<WIDTH>(data, m, riskBlocks, block, blocks, state) ||
        !SpsvAllocateHeadCertificate<WIDTH>(m, blocks, state)) {
        return riskBlocks;
    }
    asc_vf_call<SpsvTailPrepareDefects>(dim3{256, 1, 1}, data, m, 0, block, blocks, state.inverses, state.defects);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    // Narrow affine state.states favor shorter inverse chains; wide state.states favor fewer border rounds.
    // The state.local inverse keeps its 32-column storage pitch for either partition.
    constexpr int32_t groupRows = WIDTH == 4 ? 16 : 32;
    asc_vf_call<SpsvHeadInverse<groupRows>>(dim3{256, 1, 1}, data, m, state.head, block, blocks, state.inverses,
                                            state.defects, state.local);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvApplyBorder<groupRows>(data, m, 0, state.head, block, blocks, state.inverses, state.defects, state.local,
                               state.constants, state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvHeadBodyPrepare<WIDTH>>(dim3{256, 1, 1}, data, m, state.head, block, blocks,
                                            state.tree + state.leaves * (WIDTH * (WIDTH + 1) * kSpsvTriplePrefixWords),
                                            state.inverses, state.defects, state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRunSignedCertificateTree<WIDTH, SHORT_PRODUCT>(state.tree, state.states, state.leaves, state.count, block,
                                                       blocks);
    asc_vf_call<SpsvHeadBodyResults<WIDTH>>(dim3{128, 1, 1}, state.tree, state.states, state.leaves, state.head, m,
                                            block, blocks, state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return SpsvPublishBorderResult(data, m, block, blocks, state);
}

#endif // SPSV_COMPLEX_HEAD_CERTIFICATE_H_
