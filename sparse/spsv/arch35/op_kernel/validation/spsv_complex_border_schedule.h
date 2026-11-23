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

#ifndef SPSV_COMPLEX_BORDER_SCHEDULE_H_
#define SPSV_COMPLEX_BORDER_SCHEDULE_H_

#include "../common/spsv_complex_workspace.h"

#include "../../shared/spsv_workspace_layout.h"

#include "spsv_complex_certificate_schedule.h"
#include "spsv_complex_border.h"
#include "spsv_preconditioned_border.h"

struct SpsvBorderCertificateState {
    int64_t limit;
    int64_t head;
    int64_t count;
    int64_t leaves;
    int64_t tail;
    int32_t n;
    uint32_t slots;
    __gm__ uint32_t* flags;
    __gm__ float* tree;
    __gm__ float* states;
    __gm__ float* corrections;
    __gm__ float* inverses;
    __gm__ float* defects;
    __gm__ float* constants;
    __gm__ float* matrix;
    __gm__ float* qmatrix;
    __gm__ float* metrics;
    __gm__ float* residuals;
    __gm__ float* local;
};

__aicore__ inline uint32_t SpsvClassifiedCertificateEdge(__gm__ uint32_t* flags, int32_t blocks)
{
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRefreshComplexCertificateFlags(flags);
    uint32_t edge = 0;
    for (int32_t b = 0; b < blocks; ++b) {
        edge = edge > flags[b] ? edge : flags[b];
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return edge;
}

__aicore__ inline bool SpsvBeginBorderSelection(const SpsvComplexCertificateData& data, int64_t m,
                                               uint32_t riskBlocks, int32_t blocks,
                                               SpsvBorderCertificateState& state)
{
    __gm__ float* wide = (__gm__ float*)data.wide;
    __gm__ uint32_t* risk = SpsvComplexScratchView{wide, {static_cast<uint64_t>(m)}}.risk;
    if (SpsvReadCertificateRisk(risk, riskBlocks) == 0U) {
        return false;
    }
    state.slots = riskBlocks > static_cast<uint32_t>(blocks) ? riskBlocks : static_cast<uint32_t>(blocks);
    state.flags = SpsvCertificateScratchView{{wide, {static_cast<uint64_t>(m)}}, state.slots,
                                             static_cast<uint32_t>(blocks)}.flags;
    return true;
}

template <int32_t WIDTH>
__aicore__ inline SpsvBorderCertificateState SpsvMakeBorderArena(
    int64_t m, int64_t count, int64_t span, int32_t blocks, const SpsvBorderCertificateState& state)
{
    auto arena = state;
    arena.count = count;
    arena.leaves = SpsvCertificateLeafCount(count);
    arena.tree = (__gm__ float*)(state.flags + blocks);
    arena.states = arena.tree + 2 * arena.leaves * (WIDTH * (WIDTH + 1) * kSpsvTriplePrefixWords);
    arena.corrections = arena.states + 2 * arena.leaves * (WIDTH * kSpsvTriplePrefixWords);
    arena.inverses = arena.corrections + 7 * m;
    arena.defects = arena.inverses + 7 * span;
    arena.constants = arena.defects + 7 * span;
    return arena;
}

// Each independent row of a small border is distributed across cores.
// Stream-local VF barriers publish constants, then completed corrections.
template <int32_t GROUP_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvBorderConstants(
    const SpsvComplexCertificateData data, int64_t m, int64_t begin, int64_t end, int64_t group, int32_t block,
    int32_t blocks, __gm__ const float* inverses, __gm__ const float* defects, __gm__ float* constants,
    __gm__ const float* corrections)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)corrections);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int32_t row = block + warp * blocks; row < GROUP_ROWS; row += 8 * blocks) {
        int64_t at = begin + GROUP_ROWS * group + row, r = data.forward ? at : m - 1 - at;
        auto z = SpsvTriplePrefixZero();
        if (at < end) {
            for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
                int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
                if (source >= begin + GROUP_ROWS * group) {
                    continue;
                }
                z = SpsvTriplePrefixAdd(
                    z, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag,
                                                        SpsvTriplePrefixLoad(corrections + source * 7)));
            }
        }
        z = SpsvTailWarpSum(z, lane);
        if (lane == 0) {
            if (at < end) {
                z = SpsvTriplePrefixAdd(SpsvTriplePrefixLoad(defects + (at - begin) * 7),
                                        SpsvTriplePrefixMul(z, SpsvTriplePrefixLoad(inverses + (at - begin) * 7)));
            }
            SpsvTriplePrefixStore(constants + (GROUP_ROWS * group + row) * 7, z);
        }
    }
    asc_threadfence();
}

template <int32_t GROUP_ROWS>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvBorderRows(int64_t begin, int64_t end, int64_t group,
                                                                         int32_t block, int32_t blocks,
                                                                         __gm__ const float* local,
                                                                         __gm__ const float* constants,
                                                                         __gm__ float* corrections)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)constants);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int32_t row = block + warp * blocks; row < GROUP_ROWS; row += 8 * blocks) {
        int64_t at = begin + GROUP_ROWS * group + row;
        auto z = SpsvTriplePrefixZero();
        if (at < end && lane <= row) {
            z = SpsvTriplePrefixMul(SpsvTriplePrefixLoad(local + ((group * GROUP_ROWS + row) * 32 + lane) * 7),
                                    SpsvTriplePrefixLoad(constants + (GROUP_ROWS * group + lane) * 7));
        }
        z = SpsvTailWarpSum(z, lane);
        if (at < end && lane == 0) {
            SpsvTriplePrefixStore(corrections + at * 7, z);
        }
    }
    asc_threadfence();
}

template <int32_t GROUP_ROWS = 32>
__aicore__ inline void SpsvApplyBorder(const SpsvComplexCertificateData& data, int64_t m, int64_t begin, int64_t end,
                                       int32_t block, int32_t blocks, __gm__ const float* inverses,
                                       __gm__ const float* defects, __gm__ const float* local, __gm__ float* constants,
                                       __gm__ float* corrections)
{
    for (int64_t group = 0; group < (end - begin + GROUP_ROWS - 1) / GROUP_ROWS; ++group) {
        asc_vf_call<SpsvBorderConstants<GROUP_ROWS>>(dim3{256, 1, 1}, data, m, begin, end, group, block, blocks,
                                                     inverses, defects, constants, corrections);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
        asc_vf_call<SpsvBorderRows<GROUP_ROWS>>(dim3{256, 1, 1}, begin, end, group, block, blocks, local, constants,
                                                corrections);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
}

template <int32_t WIDTH, bool PRECONDITIONED>
__aicore__ inline bool SpsvSelectTailLayout(const SpsvComplexCertificateData& data, int64_t m, uint32_t riskBlocks,
                                                 int32_t block, int32_t blocks, SpsvBorderCertificateState& state)
{
    if (!SpsvBeginBorderSelection(data, m, riskBlocks, blocks, state)) {
        return false;
    }
    uint32_t slots = state.slots;
    __gm__ uint32_t* flags = state.flags;
    asc_vf_call<SpsvTailClassify<WIDTH>>(dim3{256, 1, 1}, data, m, block, blocks, flags);
    uint32_t edge = SpsvClassifiedCertificateEdge(flags, blocks);
    if (edge == 0 || (edge & 0x80000000U)) {
        return false;
    }
    int64_t limit = (m - edge) / WIDTH * WIDTH;
    if constexpr (PRECONDITIONED) {
        if (limit < 128 || m - limit < 128 || m - limit > 320 || m > 8192) {
            return false;
        }
    } else if (limit < 128 || m - limit > 512) {
        return false;
    }
    state.limit = limit;
    state.slots = slots;
    state.flags = flags;
    return true;
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvSelectTailCertificate(const SpsvComplexCertificateData& data, int64_t m, uint32_t riskBlocks,
                                                 int32_t block, int32_t blocks, SpsvBorderCertificateState& state)
{
    return SpsvSelectTailLayout<WIDTH, false>(data, m, riskBlocks, block, blocks, state);
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvAllocateTailCertificate(int64_t m, int32_t blocks, SpsvBorderCertificateState& state)
{
    int64_t tail = ((m - state.limit + 31) / 32) * 32;
    auto arena = SpsvMakeBorderArena<WIDTH>(m, state.limit / WIDTH, tail, blocks, state);
    arena.tail = tail;
    arena.local = arena.constants + 7 * tail;
    uint64_t required = SpsvComplexWorkspaceGeometry{static_cast<uint64_t>(m)}.riskWord + state.slots + blocks +
                        2 * arena.leaves * (WIDTH * (WIDTH + 2) * kSpsvTriplePrefixWords) + 7 * m + 3 * 7 * tail +
                        tail * 32 * 7;
    if (!SpsvCertificateArenaFits(m, required)) {
        return false;
    }
    state = arena;
    return true;
}

template <int32_t WIDTH, bool SHORT_PRODUCT>
__aicore__ inline void SpsvPrepareTailCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                 int32_t block, int32_t blocks,
                                                 const SpsvBorderCertificateState& state)
{
    asc_vf_call<SpsvTailPrepare<WIDTH>>(dim3{256, 1, 1}, data, m, state.limit, block, blocks,
                                        state.tree + state.leaves * (WIDTH * (WIDTH + 1) * kSpsvTriplePrefixWords));
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRunSignedCertificateTree<WIDTH, SHORT_PRODUCT>(state.tree, state.states, state.leaves, state.count, block,
                                                       blocks);
    asc_vf_call<SpsvTailHeadResults<WIDTH>>(dim3{128, 1, 1}, state.tree, state.states, state.leaves, state.limit, block,
                                            blocks, state.corrections);
    asc_vf_call<SpsvTailPrepareDefects>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.inverses,
                                        state.defects);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
}

__aicore__ inline uint32_t SpsvPublishBorderResult(const SpsvComplexCertificateData& data, int64_t m,
                                                  int32_t block, int32_t blocks,
                                                  const SpsvBorderCertificateState& state)
{
    __gm__ float* wide = (__gm__ float*)data.wide;
    __gm__ uint32_t* risk = SpsvComplexScratchView{wide, {static_cast<uint64_t>(m)}}.risk;
    asc_vf_call<SpsvTailPublish>(dim3{128, 1, 1}, state.corrections, (__gm__ SpsvComplex64*)data.vecY, wide, m,
                                 data.forward, block, blocks, risk);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return static_cast<uint32_t>(blocks);
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__aicore__ inline uint32_t SpsvRunTailCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                  uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvBorderCertificateState state{};
    if (!SpsvSelectTailCertificate<WIDTH>(data, m, riskBlocks, block, blocks, state) ||
        !SpsvAllocateTailCertificate<WIDTH>(m, blocks, state)) {
        return riskBlocks;
    }
    SpsvPrepareTailCertificate<WIDTH, SHORT_PRODUCT>(data, m, block, blocks, state);
    if (m - state.limit <= 128) {
        asc_vf_call<SpsvTailInverseParallelColumns>(dim3{32, 1, 1}, data, m, state.limit, block, blocks, state.inverses,
                                                    state.local);
    } else {
        asc_vf_call<SpsvTailInverse>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.inverses,
                                     state.defects, state.local);
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvApplyBorder(data, m, state.limit, m, block, blocks, state.inverses, state.defects, state.local, state.constants,
                    state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return SpsvPublishBorderResult(data, m, block, blocks, state);
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvSelectPreconditionedTail(const SpsvComplexCertificateData& data, int64_t m,
                                                    uint32_t riskBlocks, int32_t block, int32_t blocks,
                                                    SpsvBorderCertificateState& state)
{
    return SpsvSelectTailLayout<WIDTH, true>(data, m, riskBlocks, block, blocks, state);
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvAllocatePreconditionedTail(int64_t m, int32_t blocks, SpsvBorderCertificateState& state)
{
    int64_t tail = ((m - state.limit + 31) / 32) * 32;
    auto arena = SpsvMakeBorderArena<WIDTH>(m, state.limit / WIDTH, tail, blocks, state);
    arena.tail = tail;
    arena.n = static_cast<int32_t>(m - state.limit);
    arena.matrix = arena.constants + 7 * tail;
    arena.qmatrix = arena.matrix + 2 * tail * tail;
    arena.metrics = arena.qmatrix + tail * tail;
    arena.residuals = arena.metrics + 3 * blocks;
    uint64_t required = SpsvComplexWorkspaceGeometry{static_cast<uint64_t>(m)}.riskWord + state.slots + blocks +
                        2 * arena.leaves * (WIDTH * (WIDTH + 2) * kSpsvTriplePrefixWords) + 7 * m + 3 * 7 * tail +
                        3 * tail * tail + 3 * blocks + tail;
    if (!SpsvCertificateArenaFits(m, required)) {
        return false;
    }
    state = arena;
    return true;
}

__aicore__ inline float SpsvCheckPreconditionedTailInverse(const SpsvComplexCertificateData& data, int64_t m,
                                                           int32_t block, int32_t blocks,
                                                           const SpsvBorderCertificateState& state)
{
    asc_vf_call<SpsvPreconditionInverse>(dim3{32, 1, 1}, data, m, state.limit, block, blocks, state.matrix);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvPreconditionCheck>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.matrix,
                                       state.qmatrix);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvPreconditionNorms>(dim3{32, 1, 1}, state.n, block, blocks, state.matrix, state.qmatrix,
                                       state.metrics);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRefreshComplexCertificateFlags((__gm__ uint32_t*)state.metrics);
    float q = 0.f, inverseNorm = 0.f;
    for (int32_t b = 0; b < blocks; ++b) {
        float rowQ = state.metrics[3 * b], rowNorm = state.metrics[3 * b + 1];
        rowQ = rowQ >= 0.f && rowQ <= 3.4028235e38f ? rowQ : __builtin_inff();
        rowNorm = rowNorm >= 0.f && rowNorm <= 3.4028235e38f ? rowNorm : __builtin_inff();
        q = q > rowQ ? q : rowQ;
        inverseNorm = inverseNorm > rowNorm ? inverseNorm : rowNorm;
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    if (!(q < 0.75f) || !(inverseNorm >= 0.f && inverseNorm <= 3.4028235e38f)) {
        return kSpsvRejectedInverseNorm;
    }
    return inverseNorm;
}

__aicore__ inline void SpsvCorrectPreconditionedTail(const SpsvComplexCertificateData& data, int64_t m, int32_t block,
                                                     int32_t blocks, float inverseNorm,
                                                     const SpsvBorderCertificateState& state)
{
    __gm__ float* wide = (__gm__ float*)data.wide;
    __gm__ uint32_t* risk = SpsvComplexScratchView{wide, {static_cast<uint64_t>(m)}}.risk;
    asc_vf_call<SpsvPreconditionConstants>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.defects,
                                           state.inverses, state.corrections, state.constants);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvPreconditionPropose>(dim3{256, 1, 1}, m, state.limit, block, blocks, state.matrix, state.constants,
                                         state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvPreconditionResidual>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.constants,
                                          state.corrections, state.residuals);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvPreconditionRadius>(dim3{128, 1, 1}, m, state.limit, block, blocks, state.residuals, inverseNorm,
                                        state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvTailPublish>(dim3{128, 1, 1}, state.corrections, (__gm__ SpsvComplex64*)data.vecY, wide, m,
                                 data.forward, block, blocks, risk);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
}

template <int32_t WIDTH = kSpsvTriplePrefixWidth, bool SHORT_PRODUCT = false>
__aicore__ inline uint32_t SpsvRunPreconditionedTailCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                                uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvBorderCertificateState state{};
    if (!SpsvSelectPreconditionedTail<WIDTH>(data, m, riskBlocks, block, blocks, state)) {
        return riskBlocks;
    }
    if (!SpsvAllocatePreconditionedTail<WIDTH>(m, blocks, state)) {
        return riskBlocks;
    }
    SpsvPrepareTailCertificate<WIDTH, SHORT_PRODUCT>(data, m, block, blocks, state);
    float inverseNorm = SpsvCheckPreconditionedTailInverse(data, m, block, blocks, state);
    if (inverseNorm < 0.f) {
        return riskBlocks;
    }
    SpsvCorrectPreconditionedTail(data, m, block, blocks, inverseNorm, state);
    return static_cast<uint32_t>(blocks);
}

__aicore__ inline bool SpsvSelectFullCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                 uint32_t riskBlocks, int32_t block, int32_t blocks,
                                                 SpsvBorderCertificateState& state)
{
    if (!SpsvBeginBorderSelection(data, m, riskBlocks, blocks, state)) {
        return false;
    }
    uint32_t slots = state.slots;
    __gm__ uint32_t* flags = state.flags;
    asc_vf_call<SpsvTailClassify<>>(dim3{256, 1, 1}, data, m, block, blocks, flags);
    uint32_t edge = SpsvClassifiedCertificateEdge(flags, blocks);
    if (edge & 0x80000000U) {
        return false;
    }
    state.limit = 0;
    state.slots = slots;
    state.flags = flags;
    return true;
}

__aicore__ inline uint32_t SpsvRunFullCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                  uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvBorderCertificateState state{};
    if (!SpsvSelectFullCertificate(data, m, riskBlocks, block, blocks, state) ||
        !SpsvAllocateTailCertificate<kSpsvTriplePrefixWidth>(m, blocks, state)) {
        return riskBlocks;
    }
    SpsvPrepareTailCertificate<kSpsvTriplePrefixWidth, false>(data, m, block, blocks, state);
    asc_vf_call<SpsvTailInverse>(dim3{256, 1, 1}, data, m, state.limit, block, blocks, state.inverses,
                                 state.defects, state.local);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<SpsvTailApply>(dim3{256, 1, 1}, data, m, state.limit, block, state.inverses, state.defects,
                               state.local, state.constants, state.corrections);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return SpsvPublishBorderResult(data, m, block, blocks, state);
}

#endif // SPSV_COMPLEX_BORDER_SCHEDULE_H_
