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

#ifndef SPSV_NUMERICAL_STATE_H_
#define SPSV_NUMERICAL_STATE_H_

constexpr float kSpsvInvalidDependencyRadius = -1.0f;
constexpr float kSpsvCertifiedOverflowRadius = -2.0f;
constexpr float kSpsvUncertifiedInverseError = -1.0f;
constexpr float kSpsvRejectedInverseNorm = -1.0f;

__simt_callee__ inline bool SpsvRadiusHasTerminalState(float radius)
{
    // Preserve the legacy negative-state rule, including both named states.
    return radius < 0.0f;
}

#endif // SPSV_NUMERICAL_STATE_H_
