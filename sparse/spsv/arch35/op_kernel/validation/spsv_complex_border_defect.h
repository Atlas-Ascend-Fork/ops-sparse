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

#ifndef SPSV_COMPLEX_BORDER_DEFECT_H_
#define SPSV_COMPLEX_BORDER_DEFECT_H_

#include "spsv_complex_triple_prefix.h"

// Certified banded prefix with a bounded arbitrary border.
constexpr int32_t kSpsvTailWidth = 32;

// A complete warp validates each row, including long border rows.
template <int32_t WIDTH = kSpsvTriplePrefixWidth>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTailClassify(const SpsvComplexCertificateData data,
                                                                           int64_t m, int32_t block, int32_t blocks,
                                                                           __gm__ uint32_t* flags)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.wide);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    uint32_t edge = 0;
    for (int64_t r = static_cast<int64_t>(block) * 8 + warp; r < m; r += static_cast<int64_t>(blocks) * 8) {
        bool bad = !SpsvCertificateRowFinite(data, r, 65536);
        bad |= !SpsvCertificateInverseFinite(data, m, r);
        bool wide = data.rowPtr[r + 1] - data.rowPtr[r] > 64;
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t distance = data.forward ? r - data.colInd[p] : static_cast<int64_t>(data.colInd[p]) - r;
            wide |= distance > WIDTH;
            bad |= (p > data.rowPtr[r] && data.colInd[p] <= data.colInd[p - 1]);
            bad |= !SpsvBlockFiniteNormal(data.values[p].real) || !SpsvBlockFiniteNormal(data.values[p].imag);
        }
        if (bad) {
            edge = 0x80000000U;
        } else if (wide) {
            uint32_t at = static_cast<uint32_t>(m - (data.forward ? r : m - 1 - r));
            edge = edge > at ? edge : at;
        }
    }
    edge = asc_reduce_max(edge);
    __ubuf__ uint32_t warpEdges[8];
    if (lane == 0) {
        warpEdges[warp] = edge;
    }
    asc_syncthreads();
    if (threadIdx.x == 0) {
        for (int32_t w = 1; w < 8; ++w) {
            edge = edge > warpEdges[w] ? edge : warpEdges[w];
        }
        flags[block] = edge;
    }
    asc_threadfence();
}

__simt_callee__ inline void SpsvTailWarpExpansion(SpsvDefectExpansion& value, int32_t lane)
{
    for (int32_t mask = 16; mask > 0; mask /= 2) {
        float a = asc_shfl_xor(value.word[0], mask), b = asc_shfl_xor(value.word[1], mask);
        float c = asc_shfl_xor(value.word[2], mask), d = asc_shfl_xor(value.word[3], mask);
        float discarded = asc_shfl_xor(value.discarded, mask);
        if (lane < mask) {
            value.discarded = SpsvFloatBoundAdd(value.discarded, discarded);
            SpsvDefectExpansionAdd(value, d);
            SpsvDefectExpansionAdd(value, c);
            SpsvDefectExpansionAdd(value, b);
            SpsvDefectExpansionAdd(value, a);
        }
    }
}

__simt_callee__ inline SpsvComplex64 SpsvCertificateDiagonal(const SpsvComplexCertificateData& data, int64_t row)
{
    return data.nonUnit && data.diagPtr[row] >= 0 ?
               SpsvComplex64{data.values[data.diagPtr[row]].real, data.values[data.diagPtr[row]].imag} :
               SpsvComplex64{1.f, 0.f};
}

__simt_callee__ inline bool SpsvTailSeedDefect(const SpsvComplexCertificateData& data, int64_t r,
                                               SpsvDefectExpansion& re, SpsvDefectExpansion& im, bool valid)
{
    valid = SpsvDefectExpansionProduct(re, data.alpha.real, data.rhs[r].real) && valid;
    valid = SpsvDefectExpansionProduct(re, -data.alpha.imag, data.rhs[r].imag) && valid;
    valid = SpsvDefectExpansionProduct(im, data.alpha.real, data.rhs[r].imag) && valid;
    valid = SpsvDefectExpansionProduct(im, data.alpha.imag, data.rhs[r].real) && valid;
    SpsvComplex64 diagonal = SpsvCertificateDiagonal(data, r);
    valid = SpsvComplexDefectAccumulate(re, im, data, static_cast<int32_t>(r), diagonal) && valid;
    return valid;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvTailPrepareDefects(const SpsvComplexCertificateData data,
                                                                                 int64_t m, int64_t limit,
                                                                                 int32_t block, int32_t blocks,
                                                                                 __gm__ float* inverses,
                                                                                 __gm__ float* defects)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.wide);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t at = limit + static_cast<int64_t>(block) * 8 + warp; at < m; at += static_cast<int64_t>(blocks) * 8) {
        int64_t r = data.forward ? at : m - 1 - at;
        SpsvDefectExpansion re{{0.f, 0.f, 0.f, 0.f}, 0.f}, im{{0.f, 0.f, 0.f, 0.f}, 0.f};
        bool valid = data.rowPtr[r + 1] - data.rowPtr[r] <= 65536 && (!data.nonUnit || data.diagPtr[r] >= 0);
        if (lane == 0) {
            valid = SpsvTailSeedDefect(data, r, re, im, valid);
        }
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int32_t c = data.colInd[p];
            if (data.forward ? c < r : c > r) {
                valid = SpsvComplexDefectAccumulate(re, im, data, c, {data.values[p].real, data.values[p].imag}) &&
                        valid;
            }
        }
        uint32_t invalid = asc_reduce_max(static_cast<uint32_t>(!valid));
        SpsvTailWarpExpansion(re, lane);
        SpsvTailWarpExpansion(im, lane);
        if (lane == 0) {
            SpsvTriplePrefixValue z = SpsvTriplePrefixZero();
            float lost = 0.f;
            for (int32_t j = 3; j >= 0; --j) {
                SpsvTriplePrefixAcc(z.re, re.word[j], lost);
                SpsvTriplePrefixAcc(z.im, im.word[j], lost);
            }
            float n = static_cast<float>(data.rowPtr[r + 1] - data.rowPtr[r] + 36);
            z.radius = invalid ? __builtin_inff() :
                                 SpsvFloatBoundUp((re.discarded + im.discarded + lost) * (1.f + n * 0x1p-16f) +
                                                  n * 0x1p-110f);
            auto inverse = SpsvTriplePrefixVerifiedInverse(data, m, r);
            SpsvTriplePrefixStore(inverses + (at - limit) * 7, inverse);
            SpsvTriplePrefixStore(defects + (at - limit) * 7, SpsvTriplePrefixMul(z, inverse));
        }
    }
    asc_threadfence();
}

#endif // SPSV_COMPLEX_BORDER_DEFECT_H_
