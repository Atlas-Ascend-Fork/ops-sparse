/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_COMPLEX_DISTRIBUTED_ROW_H_
#define SPSV_COMPLEX_DISTRIBUTED_ROW_H_

#include "../common/spsv_complex_workspace.h"

// Dispatch hints only; the available block count comes from the device.
constexpr int32_t kSpsvDistributedRowThreads = 256;
constexpr int32_t kSpsvDistributedPartialWords = 8;
constexpr int32_t kSpsvDistributedMinRowLength = 1024;
constexpr int32_t kSpsvDistributedMaxLevels = 64;

template <typename RowPtrT, typename ColIndT, bool FORWARD>
__simt_vf__ __aicore__ __launch_bounds__(kSpsvDistributedRowThreads) inline void SpsvDistributedRowPartial(
    __gm__ const RowPtrT* rp, __gm__ const ColIndT* ci, __gm__ const SpsvComplex64* values,
    __gm__ const float* wide, __gm__ const float* prep, __gm__ float* partials, int32_t row, int32_t block,
    int32_t blocks)
{
    SpsvInvalidateSolveCache((__gm__ void*)prep);
    int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid % 32;
    int64_t meta = 18 * static_cast<int64_t>(row);
    RowPtrT start = rp[row], end = rp[row + 1];
    bool safe = end - start <= (1 << 20) && prep[meta + 8] > 0.f && prep[meta + 15] >= 0.f;
    float rh = 0.f, rl = 0.f, ih = 0.f, il = 0.f, er = 0.f, ei = 0.f, dependency = 0.f;
    uint32_t flags = safe ? 0U : 2U;
    if (safe) {
        SpsvAccumulateFastComplex<RowPtrT, ColIndT, FORWARD>(
            ci, values, wide, prep, start, end, block * kSpsvDistributedRowThreads + tid,
            blocks * kSpsvDistributedRowThreads, row, rh, rl, ih, il, er, ei, dependency, safe, flags);
    }
    SpsvDF re = SpsvTwoSum(rh, rl), im = SpsvTwoSum(ih, il);
    SpsvMergeFastComplexGroup<true, 32>(static_cast<int64_t>(end) - start, re, im, er, ei, dependency, safe,
                                        flags);
    SpsvMergeFastComplexBlock(lane, re, im, er, ei, dependency, safe, flags);
    if (tid == 0) {
        auto p = partials + kSpsvDistributedPartialWords * block;
        p[0] = re.hi;
        p[1] = re.lo;
        p[2] = im.hi;
        p[3] = im.lo;
        p[4] = er;
        p[5] = ei;
        p[6] = dependency;
        p[7] = static_cast<float>(flags | (safe ? 0U : 2U));
    }
    asc_threadfence();
}

__simt_callee__ inline void SpsvDistributedMerge(__gm__ const float* partials, int32_t blocks, SpsvDF& re,
                                                 SpsvDF& im, float& er, float& ei, float& dependency,
                                                 uint32_t& flags)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    for (int32_t at = lane; at < blocks; at += 32) {
        auto p = partials + kSpsvDistributedPartialWords * at;
        re = SpsvMergeEnclosed(re, {p[0], p[1]}, er);
        im = SpsvMergeEnclosed(im, {p[2], p[3]}, ei);
        er += p[4];
        ei += p[5];
        dependency += p[6];
        flags |= static_cast<uint32_t>(p[7]);
    }
    re = SpsvWarpEnclosed(re, er);
    im = SpsvWarpEnclosed(im, ei);
    dependency = asc_reduce_add(dependency);
    flags = asc_reduce_max(flags);
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT>
__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvDistributedRowPublish(
    __gm__ const RowPtrT* rp, __gm__ const ColIndT* ci, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* y, __gm__ float* wide, __gm__ const float* prep,
    __gm__ const float* partials, SpsvComplex64 alpha, int32_t row, int32_t blocks)
{
    SpsvInvalidateSolveCache((__gm__ void*)partials);
    SpsvDF re{0.f, 0.f}, im{0.f, 0.f};
    float er = 0.f, ei = 0.f, dependency = 0.f;
    uint32_t flags = 0U;
    SpsvDistributedMerge(partials, blocks, re, im, er, ei, dependency, flags);
    int64_t meta = 18 * static_cast<int64_t>(row);
    re = SpsvMergeEnclosed(re, {prep[meta + 2], prep[meta + 3]}, er);
    im = SpsvMergeEnclosed(im, {prep[meta + 4], prep[meta + 5]}, ei);
    er += prep[meta + 9] * 0x1p-23f;
    ei += prep[meta + 10] * 0x1p-23f;
    if ((flags & 1U) != 0U) {
        if (threadIdx.x == 0) {
            SpsvPublishFastComplexNaN(y, wide, prep, row, meta, flags);
        }
    } else if (SpsvCertifyFastComplex<NON_UNIT>(prep, meta, prep[meta + 15], re, im, er, ei, flags == 0U)) {
        if (threadIdx.x == 0) {
            SpsvPublishFastComplex<NON_UNIT>(y, wide, prep, row, meta,
                                             static_cast<int64_t>(rp[row + 1]) - rp[row], re, im, er, ei,
                                             dependency);
        }
    } else {
        SpsvFinishComplexColdRow<RowPtrT, ColIndT, FORWARD, NON_UNIT, true>(rp, ci, values, rhs, y, wide,
                                                                            alpha, row, prep);
    }
    asc_threadfence();
}

template <typename RowPtrT, typename ColIndT, bool FORWARD, bool NON_UNIT>
__aicore__ inline void SpsvLaunchDistributedRow(__gm__ const RowPtrT* rp, __gm__ const ColIndT* ci,
                                                __gm__ const SpsvComplex64* values,
                                                __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* y,
                                                __gm__ float* wide, __gm__ const float* prep,
                                                SpsvComplex64 alpha, int64_t m, int32_t row, int32_t blocks)
{
    // Before recovery, the exact scratch arena has no live values. Eight
    // floats per block fit in its existing three-words-per-row prefix.
    __gm__ float* partials = SpsvComplexScratchView{wide, {static_cast<uint64_t>(m)}}.arena;
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
    asc_vf_call<SpsvDistributedRowPartial<RowPtrT, ColIndT, FORWARD>>(
        dim3{kSpsvDistributedRowThreads, 1, 1}, rp, ci, values, wide, prep, partials, row, block, blocks);
    SpsvSyncSolveBlocks();
    if (block == 0) {
        asc_vf_call<SpsvDistributedRowPublish<RowPtrT, ColIndT, FORWARD, NON_UNIT>>(
            dim3{32, 1, 1}, rp, ci, values, rhs, y, wide, prep, partials, alpha, row, blocks);
    }
    // The caller's level barrier orders the publication before scratch reuse
    // and before another block consumes the completed row.
}

template <typename RowPtrT, typename ColIndT>
__aicore__ inline bool SpsvTryLaunchDistributedRow(__gm__ const RowPtrT* rp, __gm__ const ColIndT* ci,
                                                   __gm__ const SpsvComplex64* values,
                                                   __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* y,
                                                   __gm__ float* wide, __gm__ const float* prep,
                                                   SpsvComplex64 alpha, int64_t m, int32_t row,
                                                   uint32_t blocks, bool forward, bool nonUnit)
{
    if (static_cast<int64_t>(rp[row + 1]) - rp[row] < kSpsvDistributedMinRowLength ||
        static_cast<uint64_t>(kSpsvDistributedPartialWords) * blocks > 3U * static_cast<uint64_t>(m)) {
        return false;
    }
#define SPSV_DISTRIBUTED_ROW(F, N)                                                                      \
    SpsvLaunchDistributedRow<RowPtrT, ColIndT, F, N>(rp, ci, values, rhs, y, wide, prep, alpha, m, row, \
                                                     static_cast<int32_t>(blocks))
    if (forward && nonUnit) {
        SPSV_DISTRIBUTED_ROW(true, true);
    } else if (forward) {
        SPSV_DISTRIBUTED_ROW(true, false);
    } else if (nonUnit) {
        SPSV_DISTRIBUTED_ROW(false, true);
    } else {
        SPSV_DISTRIBUTED_ROW(false, false);
    }
#undef SPSV_DISTRIBUTED_ROW
    return true;
}

#endif  // SPSV_COMPLEX_DISTRIBUTED_ROW_H_
