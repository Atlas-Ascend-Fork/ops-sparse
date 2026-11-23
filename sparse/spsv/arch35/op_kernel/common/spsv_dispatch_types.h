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

#ifndef OP_KERNEL_COMMON_SPSV_DISPATCH_TYPES_H_
#define OP_KERNEL_COMMON_SPSV_DISPATCH_TYPES_H_

// Rows per level below which extra blocks cost more in grid-wide barriers than
// they return in parallelism.  A chain has one row per level and must stay on a
// single block; a schedule with wide levels wants every block it can get.
constexpr int32_t kSpsvFloatMultiBlockMinLevelWidth = 128;
constexpr int32_t kSpsvComplexMultiBlockMinLevelWidth = 256;

// The parameter list the three complex level routines share, in one place,
// the way the analysis side already does it.
#define SPSV_COMPLEX_LEVEL_PARAMS(RowPtrT, ColIndT)                                                  \
    __gm__ const RowPtrT *rowPtr, __gm__ const ColIndT *colInd, __gm__ const SpsvComplex64 *values,  \
        __gm__ const volatile SpsvComplex64 *vecX, __gm__ SpsvComplex64 *vecY, __gm__ float *yWide,  \
        __gm__ const int32_t *levelPtr, __gm__ const int32_t *levelRow, __gm__ const float *rowPrep, \
        __gm__ const int32_t *validCount

#define SPSV_COMPLEX_LEVEL_ARGS rowPtr, colInd, values, vecX, vecY, yWide, levelPtr, levelRow, rowPrep, validCount

#endif // OP_KERNEL_COMMON_SPSV_DISPATCH_TYPES_H_
