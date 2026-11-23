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

#ifndef OP_KERNEL_VALIDATION_SPSV_DEPENDENCY_RECOVERY_H_
#define OP_KERNEL_VALIDATION_SPSV_DEPENDENCY_RECOVERY_H_

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvFloatDependencyRetry(
    __gm__ const uint8_t* rowAddress, __gm__ const uint8_t* colAddress, __gm__ const float* values,
    __gm__ const float* rhs, __gm__ float* vecY, __gm__ float* state, __gm__ const uint8_t* diagAddress, float alpha,
    int64_t m, bool forward, bool nonUnit, bool rowWide, bool colWide, bool permWide, int32_t certificateBlocks)
{
    SpsvInvalidateSolveCache((__gm__ void*)state);
    __ubuf__ uint32_t warpRisk[8];
    uint32_t risk = 0;
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    for (int64_t r = tid; r < m; r += blockDim.x) {
        risk |= SpsvFloatNeedsDependencyRetry(vecY[r], state[2 * r], state[2 * r + 1]);
    }
    if (!SpsvFloatBlockHasRisk(risk, warpRisk, tid)) {
        return;
    }
    __gm__ int32_t* scale = (__gm__ int32_t*)(state + 2 * m);
    __gm__ uint32_t* residue = (__gm__ uint32_t*)(scale + m);
    __gm__ uint16_t* digits = (__gm__ uint16_t*)(residue + m);
    int64_t capacity = (kSpsvRationalBitsPerRow * m + 512 + 15) / 16;
    SpsvBigUInt numerators[kSpsvRationalTargetBatch];
    for (int32_t i = 0; i < kSpsvRationalTargetBatch; ++i) {
        numerators[i] = {digits + static_cast<int64_t>(i) * capacity, 0};
    }
    SpsvBigUInt modulus{digits + kSpsvRationalTargetBatch * capacity, 0};
    SpsvBigUInt denominator{digits + (kSpsvRationalTargetBatch + 1) * capacity, 0};
    SpsvRationalIndexArray rowPtr{(__gm__ const int32_t*)rowAddress, (__gm__ const int64_t*)rowAddress, rowWide};
    SpsvRationalIndexArray colInd{(__gm__ const int32_t*)colAddress, (__gm__ const int64_t*)colAddress, colWide};
    SpsvRationalIndexArray diagPtr{(__gm__ const int32_t*)diagAddress, (__gm__ const int64_t*)diagAddress, permWide};
    SpsvFloatCheckInverseSummary(digits, certificateBlocks, warpRisk, tid);
    if (warpRisk[0] == 0) {
        return;
    }
    uint32_t tailRisk = SpsvFloatCheckInverseTail(vecY, state, scale, m, forward, certificateBlocks, tid);
    if (!SpsvFloatBlockHasRisk(tailRisk, warpRisk, tid)) {
        return;
    }
    SpsvFloatCheckTwoBand(rowPtr, colInd, values, diagPtr, rhs, vecY, state, alpha, m, forward, nonUnit, digits,
                          warpRisk, tid);
    if (warpRisk[0] == 0) {
        return;
    }
    if (tid != 0) {
        return;
    }
    SpsvClassifyFloatRecoveryRows(rowPtr, colInd, values, diagPtr, rhs, vecY, state, alpha, m, forward, nonUnit, scale);
    SpsvDyadicRetry(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, vecY, state, digits);
    SpsvFloatRationalRecovery(rowPtr, colInd, values, diagPtr, rhs, vecY, state, alpha, m, forward, nonUnit, scale,
                              residue, numerators, modulus, denominator);
    asc_threadfence();
}

#endif // OP_KERNEL_VALIDATION_SPSV_DEPENDENCY_RECOVERY_H_
