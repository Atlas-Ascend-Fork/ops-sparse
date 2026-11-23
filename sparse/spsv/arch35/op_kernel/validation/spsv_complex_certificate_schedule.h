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

#ifndef SPSV_COMPLEX_CERTIFICATE_SCHEDULE_H_
#define SPSV_COMPLEX_CERTIFICATE_SCHEDULE_H_

#include "../common/spsv_complex_workspace.h"

#include "../../shared/spsv_workspace_layout.h"

#include "spsv_complex_triple_prefix.h"

__aicore__ inline void SpsvRefreshComplexCertificateFlags(__gm__ uint32_t* flags)
{
    // The scalar risk scan may cache the adjacent validation flags before
    // SIMT overwrites them. Cross-core completion alone does not invalidate
    // that scalar cache line, including when only one block participates.
    AscendC::GlobalTensor<uint32_t> memory;
    memory.SetGlobalBuffer(flags);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        memory);
}

__aicore__ inline uint32_t SpsvReadCertificateRisk(__gm__ uint32_t* risk, uint32_t blocks)
{
    SpsvRefreshComplexCertificateFlags(risk);
    uint32_t any = 0U;
    for (uint32_t block = 0; block < blocks; ++block) {
        any |= risk[block];
    }
    return any;
}

__aicore__ inline int64_t SpsvCertificateLeafCount(int64_t count)
{
    int64_t leaves = 1;
    while (leaves < count) {
        leaves *= 2;
    }
    return leaves;
}

template <int32_t WIDTH, bool SHORT_PRODUCT>
__aicore__ inline void SpsvRunSignedCertificateTree(__gm__ float* tree, __gm__ float* states, int64_t leaves,
                                                    int64_t count, int32_t block, int32_t blocks)
{
    for (int64_t level = leaves / 2; level >= 1; level /= 2) {
        asc_vf_call<SpsvTripleTreeUp<WIDTH, SHORT_PRODUCT>>(dim3{512, 1, 1}, tree, leaves, count, level, block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    for (int64_t level = 1; level < leaves; level *= 2) {
        asc_vf_call<SpsvTripleTreeDown<WIDTH, SHORT_PRODUCT>>(dim3{256, 1, 1}, tree, states, leaves, count, level,
                                                              block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
}

__aicore__ inline uint32_t SpsvCollectCertificateFlags(__gm__ uint32_t* flags, int32_t blocks)
{
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    SpsvRefreshComplexCertificateFlags(flags);
    uint32_t any = 0U;
    for (int32_t b = 0; b < blocks; ++b) {
        any |= flags[b];
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return any;
}

__aicore__ inline uint32_t SpsvRunComplexPrefixCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                           uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvCertificateScratchView arena{{(__gm__ float*)data.wide, {static_cast<uint64_t>(m)}}, riskBlocks,
                                     static_cast<uint32_t>(blocks)};
    if (SpsvReadCertificateRisk(arena.workspace.risk, riskBlocks) == 0U) {
        return riskBlocks;
    }
    int64_t count = (m + kSpsvPrefixWidth - 1) / kSpsvPrefixWidth;
    __gm__ float* second = arena.storage + count * kSpsvPrefixEntries * kSpsvPrefixWords;
    asc_vf_call<SpsvPrefixPrepare>(dim3{32, 1, 1}, (__gm__ const int32_t*)data.rowPtr,
                                   (__gm__ const int32_t*)data.colInd, (__gm__ const SpsvComplex64*)data.values,
                                   (__gm__ const SpsvComplex64*)data.rhs, (__gm__ const SpsvComplex64*)data.vecY,
                                   (__gm__ const int32_t*)data.diagPtr, arena.workspace.base, data.alpha, m,
                                   data.forward, block, blocks, arena.storage, arena.flags);
    if (SpsvCollectCertificateFlags(arena.flags, blocks) != 0U) {
        return riskBlocks;
    }
    __gm__ float* input = arena.storage;
    __gm__ float* output = second;
    for (int32_t gap = 1; gap < count; gap *= 2) {
        asc_vf_call<SpsvPrefixCompose>(dim3{512, 1, 1}, input, output, count, gap, block, blocks);
        if (blocks > 1) {
            SpsvPrefixBarrier();
        }
        __gm__ float* swap = input;
        input = output;
        output = swap;
    }
    asc_vf_call<SpsvPrefixPublish>(dim3{128, 1, 1}, input, (__gm__ const SpsvComplex64*)data.vecY,
                                   arena.workspace.base, m, data.forward, block, blocks, arena.workspace.risk);
    if (blocks > 1) {
        SpsvPrefixBarrier();
    }
    return static_cast<uint32_t>(blocks);
}

template <int32_t WIDTH>
__aicore__ inline uint32_t SpsvRunComplexWidthCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                          uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    SpsvCertificateScratchView arena{{(__gm__ float*)data.wide, {static_cast<uint64_t>(m)}}, riskBlocks,
                                     static_cast<uint32_t>(blocks)};
    if (SpsvReadCertificateRisk(arena.workspace.risk, riskBlocks) == 0U) {
        return riskBlocks;
    }
    int64_t count = (m + WIDTH - 1) / WIDTH;
    int64_t leaves = SpsvCertificateLeafCount(count);
    // This scan reuses the existing rational-recovery arena. Guard its
    // complete tree/state footprint before any producer can write it.
    uint64_t required = arena.storageWord +
                        2U * leaves * ((WIDTH * (WIDTH + 1)) + WIDTH) * kSpsvTriplePrefixWords;
    if (!SpsvCertificateArenaFits(m, required)) {
        return riskBlocks;
    }
    __gm__ float* states = arena.storage + 2 * leaves * (WIDTH * (WIDTH + 1)) * kSpsvTriplePrefixWords;
    __gm__ float* leafData = arena.storage + leaves * (WIDTH * (WIDTH + 1)) * kSpsvTriplePrefixWords;
    asc_vf_call<SpsvTriplePrefixPrepare<WIDTH>>(
        dim3{256, 1, 1}, (__gm__ const int32_t*)data.rowPtr, (__gm__ const int32_t*)data.colInd,
        (__gm__ const SpsvComplex64*)data.values, (__gm__ const SpsvComplex64*)data.rhs,
        (__gm__ const SpsvComplex64*)data.vecY, (__gm__ const int32_t*)data.diagPtr, arena.workspace.base, data.alpha, m,
        data.forward, data.nonUnit, block, blocks, leafData, arena.flags);
    if (SpsvCollectCertificateFlags(arena.flags, blocks) != 0U) {
        return riskBlocks;
    }
    SpsvRunSignedCertificateTree<WIDTH, false>(arena.storage, states, leaves, count, block, blocks);
    asc_vf_call<SpsvTriplePrefixPublish<WIDTH>>(dim3{128, 1, 1}, arena.storage, states, leaves,
                                                (__gm__ SpsvComplex64*)data.vecY, arena.workspace.base, m,
                                                data.forward, block, blocks, arena.workspace.risk);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return static_cast<uint32_t>(blocks);
}

#endif // SPSV_COMPLEX_CERTIFICATE_SCHEDULE_H_
