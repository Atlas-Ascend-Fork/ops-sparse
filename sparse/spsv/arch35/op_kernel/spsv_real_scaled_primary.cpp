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

#include <cstdint>
#include "kernel_operator.h"
#include "c_api/vector_datamove/vector_datamove.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "spsv_kernel.h"
#include "common/spsv_primary_status.h"
#include "math/spsv_arith.h"
#include "common/spsv_schedule_common.h"
#define SPSV_SOFT __aicore__ __attribute__((always_inline)) inline
#define SPSV_SOFT_CLZ(value) __builtin_clzll(value)
#include "math/spsv_binary64.h"
#include "recovery/spsv_rational.h"
#include "recovery/spsv_complex_rational.h"
#include "validation/spsv_complex_scaled.h"
#include "aclsparse_host_utils.h"

__aicore__ __attribute__((always_inline)) inline uint32_t SpsvRealFloatBits(float value)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__aicore__ __attribute__((always_inline)) inline uint64_t SpsvRealProduct32(uint32_t a, uint64_t b)
{
    return spsv_soft::Multiply32<false>(a, b);
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvRealPrepareOrder(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const float* values,
    __gm__ const int32_t* permutation, __gm__ uint64_t* prepared,
    int64_t m, int64_t nnz, int32_t block, int32_t blocks)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t row = static_cast<int64_t>(block) * 8 + warp; row < m; row += blocks * 8) {
        int32_t begin = rowPtr[row], end = rowPtr[row + 1];
        if (begin < 0 || end < begin || end > nnz || end - begin > 32) {
            continue;
        }
        int32_t p = begin + lane, count = end - begin;
        int32_t key = lane < count && permutation != nullptr ? permutation[p] : lane;
        int32_t rank = 0;
        for (int32_t item = 0; item < count; ++item) {
            rank += asc_shfl(key, item, 32) < key;
        }
        if (lane < count) {
            uint32_t bits;
            float value = values[p];
            __builtin_memcpy(&bits, &value, sizeof(bits));
            prepared[begin + rank] = (static_cast<uint64_t>(colInd[p]) << 32) | bits;
        }
    }
    asc_threadfence();
}

extern "C" __global__ __aicore__ void spsv_real_binary64_prepare(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    auto permutation = tiling.opA != 0 ? (__gm__ const int32_t*)(workspace + tiling.transPermOffset) : nullptr;
    if (tiling.opA != 0 && tiling.transRowPtrOffset >= 0) {
        rowPtr = workspace + tiling.transRowPtrOffset;
        colInd = workspace + tiling.transColIndOffset;
        values = workspace + tiling.transValuesOffset;
    }
    asc_vf_call<SpsvRealPrepareOrder>(dim3{256, 1, 1}, (__gm__ const int32_t*)rowPtr,
                                      (__gm__ const int32_t*)colInd, (__gm__ const float*)values, permutation,
                                      (__gm__ uint64_t*)(workspace + tiling.yWideOffset + 8 * tiling.m * sizeof(float)),
                                      tiling.m, tiling.nnz, static_cast<int32_t>(AscendC::GetBlockIdx()),
                                      static_cast<int32_t>(AscendC::GetBlockNum()));
}

__aicore__ inline bool SpsvRealReplayAdmitted(__gm__ const int32_t* rp, __gm__ const int32_t* ci,
                                                int64_t m, int64_t nnz)
{
    if (rp[0] != 0 || rp[m] != nnz) {
        return false;
    }
    for (int64_t row = 0; row < m; ++row) {
        int32_t begin = rp[row], end = rp[row + 1];
        if (begin < 0 || end < begin || end > nnz || end - begin > 32) {
            return false;
        }
        for (int32_t p = begin; p < end; ++p) {
            if (ci[p] < 0 || ci[p] >= m) {
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline void SpsvRealStageReplay(__ubuf__ uint64_t* staged, __gm__ const uint64_t* prepared,
                                          int32_t begin, int64_t nnz)
{
    int32_t count = nnz - begin < 1024 ? static_cast<int32_t>(nnz - begin) : 1024;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    asc_copy_gm2ub_align((__ubuf__ uint32_t*)staged, (__gm__ uint32_t*)(prepared + begin), count * 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
}

__aicore__ __attribute__((always_inline)) inline void SpsvRealPublishReplay(
    __gm__ uint64_t* saved, __ubuf__ uint64_t* localSaved, __gm__ float* output, int64_t row, uint64_t sum)
{
    saved[row] = sum;
    localSaved[row] = sum;
    uint32_t bits = spsv_soft::To32(sum);
    __builtin_memcpy(&output[row], &bits, sizeof(bits));
}

extern "C" __global__ __aicore__ void spsv_real_binary64_primary(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR rhs, GM_ADDR y, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(workspace);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(state);
    auto status = (__gm__ SpsvTilingData*)workspace;
    status->primaryStatus = 0;
    SpsvResolveComplexSolveCsr(rowPtr, colInd, values, workspace, tiling);
    auto rp = (__gm__ const int32_t*)rowPtr;
    auto ci = (__gm__ const int32_t*)colInd;
    auto a = (__gm__ const float*)values;
    auto x = (__gm__ const float*)rhs;
    auto out = (__gm__ float*)y;
    auto saved = (__gm__ uint64_t*)(workspace + tiling.yWideOffset + 6 * tiling.m * sizeof(float));
    __ubuf__ uint64_t localSaved[8192];
    auto prepared = (__gm__ const uint64_t*)(workspace + tiling.yWideOffset + 8 * tiling.m * sizeof(float));
    __ubuf__ uint64_t staged[1024];
    int32_t stagedBegin = -1;
    float alpha = tiling.alphaDevicePtr ? *(__gm__ const float*)tiling.alphaDevicePtr : tiling.alpha;
    uint32_t alphaBits = SpsvRealFloatBits(alpha);
    bool forward = (tiling.fillMode == 0) == (tiling.opA == 0);
    if (!SpsvRealReplayAdmitted(rp, ci, tiling.m, tiling.nnz)) {
        return;
    }
    for (int64_t step = 0; step < tiling.m; ++step) {
        int64_t row = forward ? step : tiling.m - 1 - step;
        int32_t begin = rp[row], end = rp[row + 1];
        uint64_t sum = SpsvRealProduct32(alphaBits, spsv_soft::From32(SpsvRealFloatBits(x[row])));
        for (int32_t item = begin; item < end; ++item) {
            int32_t tileBegin = item / 1024 * 1024;
            if (stagedBegin != tileBegin) {
                SpsvRealStageReplay(staged, prepared, tileBegin, tiling.nnz);
                stagedBegin = tileBegin;
            }
            uint64_t packed = staged[item - stagedBegin];
            int32_t column = static_cast<int32_t>(packed >> 32);
            if (forward ? column < row : column > row) {
                uint64_t product = SpsvRealProduct32(static_cast<uint32_t>(packed), localSaved[column]);
                sum = spsv_soft::Sub(sum, product);
            }
        }
        SpsvRealPublishReplay(saved, localSaved, out, row, sum);
    }
    status->primaryStatus = 3;
}

void spsv_real_binary64_primary_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                   GM_ADDR workspace, const SpsvTilingData& tiling, void* stream)
{
    uint32_t blocks = GetAivCoreCount();
    blocks = blocks > 0U ? blocks : tiling.numBlocks;
    spsv_real_binary64_prepare<<<blocks, nullptr, stream>>>(rowPtr, colInd, values, workspace, tiling);
    spsv_real_binary64_primary<<<1, nullptr, stream>>>(rowPtr, colInd, values, rhs, y, workspace, tiling);
}

extern "C" __schedmode__(1) __global__ __aicore__
    void spsv_real_scaled_primary(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y, GM_ADDR workspace,
                                  const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    SpsvResolveComplexSolveCsr(rp, ci, values, workspace, tiling);
    float alpha = tiling.alphaDevicePtr ? *(__gm__ const float*)tiling.alphaDevicePtr : tiling.alpha;
    SpsvRealScaledData data{(__gm__ const int32_t*)rp,
                            (__gm__ const int32_t*)ci,
                            (__gm__ const float*)values,
                            (__gm__ const float*)rhs,
                            (__gm__ const int32_t*)(workspace + tiling.diagPtrOffset),
                            (__gm__ const float*)y,
                            (__gm__ const float*)(workspace + tiling.yWideOffset),
                            {alpha, 0.f},
                            (tiling.fillMode == 0) == (tiling.opA == 0),
                            tiling.diagType == 0};
    auto result = SpsvTryScaledComplex(data, tiling.m);
    if (AscendC::GetBlockIdx() == 0) {
        ((__gm__ SpsvTilingData*)workspace)->primaryStatus = result == SpsvScaledStatus::Certified ? 1 : 0;
    }
}

void spsv_real_scaled_primary_do(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y, GM_ADDR workspace,
                                 const SpsvTilingData& tiling, void* stream)
{
    uint32_t blocks = GetAivCoreCount();
    blocks = blocks > 0U ? blocks : tiling.numBlocks;
    spsv_real_scaled_primary<<<blocks, nullptr, stream>>>(rp, ci, values, rhs, y, workspace, tiling);
}
