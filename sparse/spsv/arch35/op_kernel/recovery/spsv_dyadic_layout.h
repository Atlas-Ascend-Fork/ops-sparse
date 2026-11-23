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
#ifndef SPSV_DYADIC_LAYOUT_H_
#define SPSV_DYADIC_LAYOUT_H_

#include <cstdint>

constexpr int32_t kSpsvDyadicLimbs = 32;

#if defined(__NPU_DEVICE__)
#define SPSV_DYADIC_LAYOUT_DEVICE __aicore__
#else
#define SPSV_DYADIC_LAYOUT_DEVICE
#endif
SPSV_DYADIC_LAYOUT_DEVICE constexpr int64_t SpsvDyadicDigitOffset(int64_t row, int32_t limb)
{
    return row * kSpsvDyadicLimbs + limb;
}

#undef SPSV_DYADIC_LAYOUT_DEVICE

#endif
