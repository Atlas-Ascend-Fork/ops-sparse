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

// Signed original-equation residual certificate; acceptance and exact recovery remain unchanged.
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "spsv_kernel.h"
#include "common/spsv_primary_status.h"
#include "validation/spsv_complex_triple_prefix.h"
#include "validation/spsv_complex_certificate_schedule.h"
#include "validation/spsv_inverse_bound.h"
#include "aclsparse_host_utils.h"

struct SpsvRealCertificateData {
    SpsvRationalIndexArray rowPtr, colInd, diagPtr;
    __gm__ const float* values;
    __gm__ const float* rhs;
    __gm__ float* y;
    __gm__ float* state;
    float alpha;
    int64_t m;
    bool forward, nonUnit;
};

__simt_callee__ inline SpsvTriplePrefixValue SpsvRealLoad(__gm__ const float* p)
{
    return {{p[0], p[1], p[2]}, {0.f, 0.f, 0.f}, p[3]};
}
__simt_callee__ inline void SpsvRealStore(__gm__ float* p, SpsvTriplePrefixValue a)
{
    p[0] = a.re.hi;
    p[1] = a.re.lo;
    p[2] = a.re.tail;
    p[3] = a.radius;
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvRealInverse(const SpsvRealCertificateData& data, int64_t r)
{
    if (!data.nonUnit) {
        return {{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    }
    int64_t dp = data.diagPtr[r];
    if (dp < 0) {
        return {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, __builtin_inff()};
    }
    float d = data.values[dp];
    if (!SpsvBlockFiniteNormal(d) || d == 0.f) {
        return {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, __builtin_inff()};
    }
    float hi = 1.f / d;
    float lo = fmaf(-d, hi, 1.f) / d;
    SpsvTriplePrefixValue q{{hi, lo, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue diagonal{{d, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    SpsvTriplePrefixValue one{{1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, 0.f};
    auto defect = SpsvTriplePrefixAdd(one, SpsvTriplePrefixNeg(SpsvTriplePrefixMul(diagonal, q)));
    defect.radius = 0.f;
    q = SpsvTriplePrefixAdd(q, SpsvTriplePrefixMul(q, defect));
    q.radius = 0.f;
    defect = SpsvTriplePrefixAdd(one, SpsvTriplePrefixNeg(SpsvTriplePrefixMul(diagonal, q)));
    q.radius = SpsvFloatBoundDiv(SpsvFloatBoundAdd(SpsvTriplePrefixNorm(defect), defect.radius), SpsvAbsF(d));
    return q;
}

__simt_callee__ inline SpsvTriplePrefixValue SpsvRealDefect(const SpsvRealCertificateData& data, int64_t r)
{
    SpsvDefectExpansion expansion{{0.f, 0.f, 0.f, 0.f}, 0.f};
    bool valid = SpsvDefectExpansionProduct(expansion, data.alpha, data.rhs[r]);
    float d = data.nonUnit ? (data.diagPtr[r] >= 0 ? data.values[data.diagPtr[r]] : 0.f) : 1.f;
    valid = SpsvDefectExpansionProduct(expansion, -d, data.y[r]) && valid;
    valid = SpsvDefectExpansionProduct(expansion, -d, data.state[2 * r]) && valid;
    for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
        int64_t c = data.colInd[p];
        if (!(data.forward ? c < r : c > r)) {
            continue;
        }
        valid = SpsvDefectExpansionProduct(expansion, -data.values[p], data.y[c]) && valid;
        valid = SpsvDefectExpansionProduct(expansion, -data.values[p], data.state[2 * c]) && valid;
    }
    auto z = SpsvTriplePrefixZero();
    float lost = 0.f;
    for (int j = 3; j >= 0; --j) {
        SpsvTriplePrefixAcc(z.re, expansion.word[j], lost);
    }
    float n = static_cast<float>(data.rowPtr[r + 1] - data.rowPtr[r] + 4);
    z.radius = valid ? SpsvFloatBoundUp((expansion.discarded + lost) * (1.f + n * 0x1p-16f) + n * 0x1p-110f) :
                       __builtin_inff();
    return z;
}

// Generated tree traversal retains the tested affine composition order;
// each stored node has three real center words plus its outward radius.
#include "validation/spsv_float_prefix_tree.h"

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvRealInitialRisk(const SpsvRealCertificateData data,
                                                                              int32_t block, int32_t blocks,
                                                                              __gm__ uint32_t* flags)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.state);
    }
    asc_syncthreads();
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    uint32_t risk = 0;
    for (int64_t r = static_cast<int64_t>(block) * 256 + tid; r < data.m; r += static_cast<int64_t>(blocks) * 256) {
        risk |= SpsvFloatNeedsDependencyRetry(data.y[r], data.state[2 * r], data.state[2 * r + 1]);
    }
    risk = asc_reduce_max(risk);
    __ubuf__ uint32_t warps[8];
    if ((tid & 31) == 0) {
        warps[tid / 32] = risk;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 8; ++w) {
            risk |= warps[w];
        }
        flags[block] = risk;
    }
    asc_threadfence();
}

template <int32_t WIDTH>
__simt_callee__ inline uint32_t SpsvRealValidateRows(const SpsvRealCertificateData& data, int32_t block, int32_t blocks,
                                                     int32_t tid)
{
    uint32_t bad = 0;
    for (int64_t r = static_cast<int64_t>(block) * 256 + tid; r < data.m; r += static_cast<int64_t>(blocks) * 256) {
        bad |= !SpsvBlockFiniteNormal(data.alpha) || !SpsvBlockFiniteNormal(data.rhs[r]) ||
               !SpsvBlockFiniteNormal(data.y[r]) || !SpsvBlockFiniteNormal(data.state[2 * r]) ||
               !(data.state[2 * r + 1] >= 0.f) || data.rowPtr[r + 1] - data.rowPtr[r] > 64;
        if (data.nonUnit) {
            auto inverse = SpsvRealInverse(data, r);
            bad |= !SpsvIsFiniteF(inverse.radius);
        }
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t c = data.colInd[p], distance = data.forward ? r - c : c - r;
            bad |= distance > WIDTH || !SpsvBlockFiniteNormal(data.values[p]);
            // Every dependency is visited explicitly; transpose scatter order
            // need not be sorted for the signed affine recurrence.
            bad |= c < 0 || c >= data.m;
        }
    }
    return bad;
}

template <int32_t WIDTH>
__simt_callee__ inline void SpsvRealPrepareRow(const SpsvRealCertificateData& data, int64_t part, int32_t row,
                                               int32_t lane, __gm__ float* leaf, __gm__ const float* inverses)
{
    int64_t at = part * WIDTH + row, r = data.forward ? at : data.m - 1 - at;
    auto z = SpsvTriplePrefixZero(), dependencies = SpsvTriplePrefixZero();
    if (at < data.m) {
        if (lane == WIDTH) {
            z = SpsvRealLoad(leaf + SpsvRealPrefixAddress<WIDTH>(part, row, WIDTH));
        }
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t c = data.colInd[p], source = data.forward ? c : data.m - 1 - c;
            if (source >= at) {
                continue;
            }
            auto value = SpsvTriplePrefixZero();
            if (source >= part * WIDTH) {
                value = SpsvRealLoad(leaf +
                                     SpsvRealPrefixAddress<WIDTH>(part, static_cast<int>(source - part * WIDTH), lane));
            } else if (source - (part - 1) * WIDTH == lane) {
                value.re.hi = 1.f;
            } else if (source < (part - 1) * WIDTH) {
                z.radius = __builtin_inff();
            }
            dependencies = SpsvTriplePrefixAdd(dependencies,
                                               SpsvTriplePrefixMulComplexScalar(-data.values[p], 0.f, value));
        }
        z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixMul(dependencies, SpsvRealLoad(inverses + at * 4)));
    }
    SpsvRealStore(leaf + SpsvRealPrefixAddress<WIDTH>(part, row, lane), z);
}

template <int32_t WIDTH>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvRealPrepare(const SpsvRealCertificateData data,
                                                                          int32_t block, int32_t blocks,
                                                                          __gm__ float* leaf, __gm__ uint32_t* invalid)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)data.state);
    }
    asc_syncthreads();
    int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    uint32_t bad = 0;
    bad = SpsvRealValidateRows<WIDTH>(data, block, blocks, tid);
    bad = asc_reduce_max(bad);
    __ubuf__ uint32_t warps[8];
    if (lane == 0) {
        warps[warp] = bad;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 8; ++w) {
            bad |= warps[w];
        }
        invalid[block] = bad;
    }
    int64_t count = (data.m + WIDTH - 1) / WIDTH;
    __gm__ float* inverses = leaf + count * WIDTH * (WIDTH + 1) * 4;
    for (int64_t first = static_cast<int64_t>(block) * 8; first < count; first += static_cast<int64_t>(blocks) * 8) {
        int64_t part = first + warp, logical = part * WIDTH + lane;
        if (part < count && lane < WIDTH) {
            auto inverse = SpsvTriplePrefixZero(), defect = SpsvTriplePrefixZero();
            if (logical < data.m) {
                int64_t r = data.forward ? logical : data.m - 1 - logical;
                inverse = SpsvRealInverse(data, r);
                defect = SpsvTriplePrefixMul(SpsvRealDefect(data, r), inverse);
            }
            SpsvRealStore(inverses + logical * 4, inverse);
            SpsvRealStore(leaf + SpsvRealPrefixAddress<WIDTH>(part, lane, WIDTH), defect);
        }
        asc_syncthreads();
        if (part >= count || lane > WIDTH) {
            continue;
        }
        for (int row = 0; row < WIDTH; ++row) {
            SpsvRealPrepareRow<WIDTH>(data, part, row, lane, leaf, inverses);
        }
    }
    asc_threadfence();
}

template <int32_t WIDTH>
__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvRealPublish(const SpsvRealCertificateData data,
                                                                          const __gm__ float* tree,
                                                                          const __gm__ float* states, int64_t leaves,
                                                                          int32_t block, int32_t blocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)tree);
    }
    asc_syncthreads();
    for (int64_t at = static_cast<int64_t>(block) * 128 + threadIdx.x; at < data.m;
         at += static_cast<int64_t>(blocks) * 128) {
        int64_t r = data.forward ? at : data.m - 1 - at;
        auto z = SpsvRealTreeResult<WIDTH>(tree, states, leaves, at);
        float bound = SpsvFloatBoundAdd(z.radius, SpsvTriplePrefixTightScalarNorm(z.re));
        if (bound >= 0.f && bound < data.state[2 * r + 1]) {
            data.state[2 * r + 1] = bound;
        }
        float high = data.y[r], low = data.state[2 * r], error = data.state[2 * r + 1];
        if (SpsvFloatNeedsDependencyRetry(high, low, error) &&
            SpsvTriplePrefixCorrectComponent(high, low, error, z.re, z.radius)) {
            data.y[r] = high;
            data.state[2 * r] = low;
            data.state[2 * r + 1] = error;
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvRealInvalidateOldCertificate(__gm__ float* state,
                                                                                           int64_t m, int32_t oldBlocks,
                                                                                           int32_t block,
                                                                                           int32_t blocks)
{
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    __gm__ float* results = state + 4 * m;
    for (int64_t i = static_cast<int64_t>(block) * 256 + tid; i < 2 * oldBlocks;
         i += static_cast<int64_t>(blocks) * 256) {
        results[i] = __builtin_inff();
    }
    for (int64_t r = static_cast<int64_t>(block) * 256 + tid; r < m; r += static_cast<int64_t>(blocks) * 256) {
        results[2 * oldBlocks + kSpsvInverseRowWords * r] = __builtin_inff();
        results[2 * oldBlocks + kSpsvInverseRowWords * r + 1] = 1.f;
    }
    asc_threadfence();
}

template <int32_t WIDTH>
__aicore__ inline void SpsvRunRealCertificateTree(__gm__ float* tree, __gm__ float* states, int64_t leaves,
                                                  int64_t count, int32_t block, int32_t blocks)
{
    for (int64_t level = leaves / 2; level >= 1; level /= 2) {
        asc_vf_call<SpsvRealTreeUp<WIDTH>>(dim3{512, 1, 1}, tree, leaves, count, level, block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    for (int64_t level = 1; level < leaves; level *= 2) {
        asc_vf_call<SpsvRealTreeDown<WIDTH>>(dim3{256, 1, 1}, tree, states, leaves, count, level, block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
}

__aicore__ inline SpsvRealCertificateData SpsvPrepareRealCertificateData(GM_ADDR rp, GM_ADDR ci, GM_ADDR values,
                                                                         GM_ADDR rhs, GM_ADDR y, GM_ADDR workspace,
                                                                         __gm__ float* state,
                                                                         const SpsvTilingData& tiling)
{
    int64_t m = tiling.m;
    float alpha = tiling.alphaDevicePtr ? *(__gm__ const float*)tiling.alphaDevicePtr : tiling.alpha;
    SpsvRealCertificateData data{{(__gm__ const int32_t*)rp, (__gm__ const int64_t*)rp, tiling.indexType != 0},
                                 {(__gm__ const int32_t*)ci, (__gm__ const int64_t*)ci, tiling.colIndType != 0},
                                 {(__gm__ const int32_t*)(workspace + tiling.diagPtrOffset),
                                  (__gm__ const int64_t*)(workspace + tiling.diagPtrOffset), tiling.permType != 0},
                                 (__gm__ const float*)values,
                                 (__gm__ const float*)rhs,
                                 (__gm__ float*)y,
                                 state,
                                 alpha,
                                 m,
                                 (tiling.fillMode == 0) == (tiling.opA == 0),
                                 tiling.diagType == 0};
    return data;
}

template <int32_t WIDTH>
__aicore__ inline bool SpsvRealCertificateFits(int64_t m, int64_t count, int64_t leaves, int32_t blocks)
{
    int64_t capacity = (kSpsvRationalBitsPerRow * m + 527) / 16;
    int64_t available = (kSpsvRationalTargetBatch + 2) * capacity / 2;
    int64_t required = 2 * leaves * WIDTH * (WIDTH + 2) * 4;
    // The inverse preparation scratch is dead before tree state writes.
    int64_t prepareEnd = (leaves + count) * WIDTH * (WIDTH + 1) * 4 + (count * WIDTH) * 4;
    if (required < prepareEnd) {
        required = prepareEnd;
    }
    return required + blocks <= available;
}

extern "C" __schedmode__(1) __global__ __aicore__
    void spsv_float_prefix_certificate_kernel(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                              GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvRealPrimaryCertified(workspace, tiling)) {
        return;
    }
    constexpr int32_t WIDTH = 12;
    if (((__gm__ const SpsvTilingData*)workspace)->numLevels <= 1) {
        return;
    }
    int64_t m = tiling.m, count = (m + WIDTH - 1) / WIDTH, leaves = 1;
    while (leaves < count) {
        leaves *= 2;
    }
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx()), blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    __gm__ float* state = (__gm__ float*)(workspace + tiling.yWideOffset) + m;
    // The light full-chain candidate deliberately skipped inverse preparation.
    // Mark those summaries invalid even when this tree cannot fit its arena.
    if (((__gm__ const SpsvTilingData*)workspace)->numLevels == m && m >= 128 && tiling.nnz <= 12 * m) {
        asc_vf_call<SpsvRealInvalidateOldCertificate>(dim3{256, 1, 1}, state, m, static_cast<int32_t>(tiling.numBlocks),
                                                      block, blocks);
    }
    if (!SpsvRealCertificateFits<WIDTH>(m, count, leaves, blocks)) {
        return;
    }
    auto data = SpsvPrepareRealCertificateData(rp, ci, values, rhs, y, workspace, state, tiling);
    __gm__ uint32_t* flags = (__gm__ uint32_t*)(state + 2 * m);
    asc_vf_call<SpsvRealInitialRisk>(dim3{256, 1, 1}, data, block, blocks, flags);
    uint32_t any = SpsvCollectCertificateFlags(flags, blocks);
    if (any == 0) {
        return;
    }
    __gm__ float* tree = state + 4 * m + blocks;
    __gm__ float* states = tree + 2 * leaves * WIDTH * (WIDTH + 1) * 4;
    asc_vf_call<SpsvRealPrepare<WIDTH>>(dim3{256, 1, 1}, data, block, blocks, tree + leaves * WIDTH * (WIDTH + 1) * 4,
                                        flags);
    uint32_t bad = SpsvCollectCertificateFlags(flags, blocks);
    if (bad == 0) {
        SpsvRunRealCertificateTree<WIDTH>(tree, states, leaves, count, block, blocks);
        asc_vf_call<SpsvRealPublish<WIDTH>>(dim3{128, 1, 1}, data, tree, states, leaves, block, blocks);
        if (blocks > 1) {
            SpsvTriplePrefixBarrier();
        }
    }
    asc_vf_call<SpsvRealInvalidateOldCertificate>(dim3{256, 1, 1}, state, m, static_cast<int32_t>(tiling.numBlocks),
                                                  block, blocks);
}

void spsv_float_prefix_certificate_kernel_do(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                             GM_ADDR workspace, const SpsvTilingData& tiling, void* stream)
{
    if (tiling.nnz > 12 * tiling.m || (tiling.numLevels != 0 && tiling.numLevels != tiling.m)) {
        return;
    }
    uint32_t blocks = GetAivCoreCount();
    if (blocks == 0) {
        blocks = tiling.numBlocks;
    }
    spsv_float_prefix_certificate_kernel<<<blocks, nullptr, stream>>>(rp, ci, values, rhs, y, workspace, tiling);
}
