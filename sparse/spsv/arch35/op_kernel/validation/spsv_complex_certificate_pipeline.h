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

#ifndef SPSV_COMPLEX_CERTIFICATE_PIPELINE_H_
#define SPSV_COMPLEX_CERTIFICATE_PIPELINE_H_

#include "../common/spsv_complex_workspace.h"
#include "spsv_complex_certificate_schedule.h"
#include "spsv_complex_border.h"
#include "spsv_complex_head_certificate.h"

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvComplexCheckCompletedRows(
    __gm__ const SpsvComplex64* vecY, __gm__ const float* wide, __gm__ uint32_t* risks, int64_t m,
    uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if (tid == 0) {
        asc_dcci_entire((__gm__ void*)wide);
    }
    asc_syncthreads();
    __gm__ const float* prep = wide + 6 * m;
    uint32_t risk = 0;
    for (int64_t r = static_cast<int64_t>(block) * blockDim.x + tid; r < m;
         r += static_cast<int64_t>(blocks) * blockDim.x) {
        if (m > 2048) {
            // A large correctly rounded FP32 result can have a wide absolute
            // enclosure. Preserve the independent rounding-cell certificate
            // before requesting recovery under the task's absolute budget.
            risk |= SpsvFloatNeedsAcceptanceRetry(vecY[r].real, wide[4 * r + 1], prep[18 * r + 9]) &&
                    SpsvFloatNeedsDependencyRetry(vecY[r].real, wide[4 * r + 1], prep[18 * r + 9]);
            risk |= SpsvFloatNeedsAcceptanceRetry(vecY[r].imag, wide[4 * r + 3], prep[18 * r + 10]) &&
                    SpsvFloatNeedsDependencyRetry(vecY[r].imag, wide[4 * r + 3], prep[18 * r + 10]);
        } else {
            risk |= SpsvFloatNeedsDependencyRetry(vecY[r].real, wide[4 * r + 1], prep[18 * r + 9]);
            risk |= SpsvFloatNeedsDependencyRetry(vecY[r].imag, wide[4 * r + 3], prep[18 * r + 10]);
        }
    }
    risk = asc_reduce_max(risk);
    __ubuf__ uint32_t warpRisk[8];
    if ((tid & 31) == 0) {
        warpRisk[tid / 32] = risk;
    }
    asc_syncthreads();
    for (uint32_t b = static_cast<uint32_t>(blocks) + static_cast<uint32_t>(block) * blockDim.x + tid;
         b < riskBlocks; b += static_cast<uint32_t>(blocks) * blockDim.x) {
        risks[b] = 0U;
    }
    if (tid == 0) {
        for (int32_t w = 1; w < 8; ++w) {
            risk |= warpRisk[w];
        }
        risks[block] = risk;
    }
    asc_threadfence();
}

__aicore__ inline void SpsvScanCompletedComplexRows(__gm__ const SpsvComplex64* y, __gm__ const float* wide,
                                                    int64_t m, uint32_t riskBlocks, int32_t block,
                                                    int32_t blocks)
{
    auto scratch = SpsvComplexScratchView{(__gm__ float*)wide, {static_cast<uint64_t>(m)}};
    asc_vf_call<SpsvComplexCheckCompletedRows>(dim3{256, 1, 1}, y, wide, scratch.risk, m, riskBlocks, block, blocks);
    // Every scan producer completes before any certificate reads or reuses
    // the risk arena. Certificate consumers also invalidate their scalar cache.
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    } else {
        // The scalar risk consumer must wait for the SIMT scan even when
        // there is no grid rendezvous. The former standalone scan launch
        // supplied this ordering through stream completion.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// A shallow level schedule propagates signed residual corrections from
// completed predecessor levels, without substituting rounded solutions.
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvLevelCertificateRows(
    const SpsvComplexCertificateData data, int64_t m, __gm__ const int32_t* levelPtr, __gm__ const int32_t* levelRow,
    int32_t level, int32_t block, int32_t blocks, __gm__ const float* inverses, __gm__ const float* defects,
    __gm__ float* corrections)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)corrections);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t i = static_cast<int64_t>(levelPtr[level]) + block * 8 + warp; i < levelPtr[level + 1];
         i += static_cast<int64_t>(blocks) * 8) {
        int64_t r = levelRow[i];
        int64_t at = data.forward ? r : m - 1 - r;
        SpsvTriplePrefixValue sum = SpsvTriplePrefixZero();
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t c = data.colInd[p];
            if (!(data.forward ? c < r : c > r)) {
                continue;
            }
            int64_t source = data.forward ? c : m - 1 - c;
            sum = SpsvTriplePrefixAdd(sum,
                                      SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag,
                                                                       SpsvTriplePrefixLoad(corrections + source * 7)));
        }
        sum = SpsvTailWarpSum(sum, lane);
        if (lane == 0) {
            sum = SpsvTriplePrefixAdd(SpsvTriplePrefixLoad(defects + at * 7),
                                      SpsvTriplePrefixMul(sum, SpsvTriplePrefixLoad(inverses + at * 7)));
            SpsvTriplePrefixStore(corrections + at * 7, sum);
        }
    }
    asc_threadfence();
}

__aicore__ inline uint32_t SpsvRunLevelCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                   __gm__ const int32_t* levelPtr, __gm__ const int32_t* levelRow,
                                                   int32_t levels, uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    __gm__ float* wide = (__gm__ float*)data.wide;
    __gm__ uint32_t* risk = SpsvComplexScratchView{wide, {static_cast<uint64_t>(m)}}.risk;
    if (SpsvReadCertificateRisk(risk, riskBlocks) == 0U) {
        return riskBlocks;
    }
    uint32_t slots = riskBlocks > static_cast<uint32_t>(blocks) ? riskBlocks : static_cast<uint32_t>(blocks);
    __gm__ float* inverses = (__gm__ float*)(risk + slots);
    __gm__ float* defects = inverses + 7 * m;
    __gm__ float* corrections = defects + 7 * m;
    uint64_t required = SpsvComplexWorkspaceGeometry{static_cast<uint64_t>(m)}.riskWord +
                        3U * kSpsvTriplePrefixWords * static_cast<uint64_t>(m) + slots;
    if (!SpsvCertificateArenaFits(m, required)) {
        return riskBlocks;
    }
    asc_vf_call<SpsvTailPrepareDefects>(dim3{256, 1, 1}, data, m, 0, block, blocks, inverses, defects);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    for (int32_t level = 0; level < levels; ++level) {
        asc_vf_call<SpsvLevelCertificateRows>(dim3{256, 1, 1}, data, m, levelPtr, levelRow, level, block, blocks,
                                              inverses, defects, corrections);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    asc_vf_call<SpsvTailPublish>(dim3{128, 1, 1}, corrections, (__gm__ SpsvComplex64*)data.vecY, wide, m, data.forward,
                                 block, blocks, risk);
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return static_cast<uint32_t>(blocks);
}

__aicore__ inline bool SpsvComplexRecoveryHasRisk(const SpsvComplexCertificateData& data, int64_t m,
                                                  uint32_t riskBlocks, int32_t blocks)
{
    auto flags = SpsvComplexScratchView{(__gm__ float*)data.wide, {static_cast<uint64_t>(m)}}.risk;
    SpsvRefreshComplexCertificateFlags(flags);
    uint32_t risk = 0U;
    for (uint32_t b = 0; b < riskBlocks; ++b) {
        risk |= flags[b];
    }
    // Every participant consumes the immutable flags before another stage
    // can reuse this arena, including when the unanimous result is no risk.
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    return risk != 0U;
}

__aicore__ inline void SpsvRunSmallPrimaryCertificate(const SpsvComplexCertificateData& data, int64_t m,
                                                      uint32_t riskBlocks, int32_t block, int32_t blocks)
{
    uint32_t count = data.nonUnit ? SpsvRunTailCertificate<4, true>(data, m, riskBlocks, block, blocks) :
                                    SpsvRunHeadCertificate<4>(data, m, riskBlocks, block, blocks);
    auto flags = SpsvComplexScratchView{(__gm__ float*)data.wide, {static_cast<uint64_t>(m)}}.risk;
    for (uint32_t at = count + static_cast<uint32_t>(block); at < riskBlocks; at += static_cast<uint32_t>(blocks)) {
        flags[at] = 0U;
    }
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
}

__aicore__ inline bool SpsvPrepareComplexRecovery(const SpsvComplexCertificateData& data, int64_t m,
                                                  uint32_t riskBlocks, int32_t block, int32_t blocks, bool fusedPrimary)
{
    if (fusedPrimary) {
        SpsvScanCompletedComplexRows(data.vecY, data.wide, m, riskBlocks, block, blocks);
    }
    if (!SpsvComplexRecoveryHasRisk(data, m, riskBlocks, blocks)) {
        return false;
    }
    // The four-row primary certificate is only validated for the original
    // large-matrix path. Small matrices fuse their completed-row scan here,
    // then retain the exact dyadic/CRT recovery decision from those flags.
    if (!fusedPrimary || m < 1024) {
        return true;
    }
    SpsvRunSmallPrimaryCertificate(data, m, riskBlocks, block, blocks);
    return SpsvComplexRecoveryHasRisk(data, m, riskBlocks, blocks);
}

// One mutable risk count follows each attempt. The phase changes only the
// applicable stages; each method preserves the prior launch sizes and barriers.
struct SpsvComplexCertificatePipeline {
    const SpsvComplexCertificateData& data;
    int64_t m;
    uint32_t risk;
    int32_t block;
    int32_t blocks;

    template <int32_t WIDTH>
    __aicore__ inline void TailPair()
    {
        risk = SpsvRunTailCertificate<WIDTH, true>(data, m, risk, block, blocks);
        risk = SpsvRunTailCertificate<WIDTH, false>(data, m, risk, block, blocks);
    }

    template <int32_t WIDTH>
    __aicore__ inline void HeadPair()
    {
        risk = SpsvRunHeadCertificate<WIDTH, true>(data, m, risk, block, blocks);
        risk = SpsvRunHeadCertificate<WIDTH, false>(data, m, risk, block, blocks);
    }

    template <int32_t WIDTH, bool WIDE_PRIMARY>
    __aicore__ inline uint32_t Run(int32_t levels, __gm__ const int32_t* levelPtr, __gm__ const int32_t* levelRow)
    {
        if constexpr (!WIDE_PRIMARY) {
            if (m < 128) {
                return risk;
            }
            if (levels > 1 && levels <= 64 && m / levels >= 32) {
                risk = SpsvRunLevelCertificate(data, m, levelPtr, levelRow, levels, risk, block, blocks);
            }
        }
        if (data.nonUnit) {
            if constexpr (WIDE_PRIMARY) {
                TailPair<WIDTH>();
                if (m <= 8192 && levels > m / 2 && levelPtr[1] - levelPtr[0] >= 128) {
                    risk = SpsvRunPreconditionedTailCertificate<16, false>(data, m, risk, block, blocks);
                }
            } else if (m >= 1024) {
                TailPair<WIDTH>();
            }
            risk = SpsvRunComplexWidthCertificate<WIDTH>(data, m, risk, block, blocks);
        } else if constexpr (WIDE_PRIMARY) {
            HeadPair<WIDTH>();
        } else {
            if (m >= 1024) {
                risk = SpsvRunHeadCertificate<WIDTH>(data, m, risk, block, blocks);
            }
            risk = SpsvRunComplexPrefixCertificate(data, m, risk, block, blocks);
            if (m >= 1024) {
                risk = SpsvRunFullCertificate(data, m, risk, block, blocks);
            }
        }
        return risk;
    }
};

#endif // SPSV_COMPLEX_CERTIFICATE_PIPELINE_H_
