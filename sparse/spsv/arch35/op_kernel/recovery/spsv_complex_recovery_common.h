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
#ifndef SPSV_COMPLEX_RECOVERY_COMMON_H_
#define SPSV_COMPLEX_RECOVERY_COMMON_H_

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "../spsv_kernel.h"
#include "../common/spsv_complex_workspace.h"
#include "../common/spsv_primary_status.h"
#include "../math/spsv_arith.h"
#include "../validation/spsv_float_bound.h"
#include "spsv_rational.h"
#include "spsv_complex_rational.h"

#include "spsv_complex_rational_retry.h"

__aicore__ inline SpsvComplex64 SpsvRecoveryAlpha(float real, float imag, uint64_t devicePtr)
{
    SpsvComplex64 alpha{real, imag};
    if (devicePtr != 0) {
        __gm__ const float* pointer = (__gm__ const float*)devicePtr;
        alpha = {pointer[0], pointer[1]};
    }
    return alpha;
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvComplexRationalRecoveryBody(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* vecY, __gm__ float* yWide, __gm__ const int32_t* diagPtr,
    SpsvComplex64 alpha, int64_t m, bool forward, bool nonUnit, int32_t riskBlocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)yWide);
    }
    asc_syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    SpsvComplexScratchView scratch{yWide, {static_cast<uint64_t>(m)}};
    const __gm__ uint32_t* continuation = scratch.continuation;
    if (continuation[0] != 1U) {
        return;
    }
    SpsvComplexRationalDependencyRetry(rowPtr, colInd, values, rhs, vecY, yWide, diagPtr, alpha, m, forward, nonUnit);
    asc_threadfence();
}

#endif // SPSV_COMPLEX_RECOVERY_COMMON_H_
