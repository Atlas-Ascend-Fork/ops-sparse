/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file matmul_kernel.h
 * \brief Matmul / epilogue / fused kernel launcher declarations
 *        (ASC side defines, CXX side includes).
 * host.cpp (CXX) includes this header and calls launchers as plain C functions.
 * kernel.cpp (ASC) defines the launchers and wraps the <<<>>> invocation.
 * host.cpp MUST NOT write <<<>>> directly.
 */

#ifndef SPLT_MATMUL_KERNEL_H
#define SPLT_MATMUL_KERNEL_H

#include <cstdint>

// Define GM_ADDR for the CXX (host) TU so the launcher
// declarations can use GM_ADDR uniformly. On the ASC (kernel) TU, GM_ADDR is
// already defined by the compiler — the #ifndef guard skips redefinition.
// This mirrors the spsm_kernel.h pattern and eliminates the (GM_ADDR) casts in
// <<<>>> calls that triggered the "cce_global attribute ignored" warning.
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

// 公共内部头（AclsparseltTilingData / aclsparseStatus_t / aclrtStream 等
// host 侧类型经其传递 include）。幂等 include（该头自带 include guard；
// 其 host-only 段经 __CCE_AICORE__ 守卫跳过）。
#include "shared/aclsparselt_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cube matmul: M(m,k) x N(k,n) -> temp (splitK*m*n, FP32).
extern void splt_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm,
    GM_ADDR tilingGm, int32_t dataType, uint32_t blockDim, void *stream);

// Epilogue: D = alpha * reduce(temp) + beta * C.
// outDataType — INT8 path selects INT8 (saturation Cast) vs INT32 (direct write).
// cDataType — C 矩阵实际 dtype（INT8 A/B 超集行下 C 可为
// INT8/INT32/FP16/BF16，CType 模板贯通分派；非 INT8 时与 dataType 同值）。
extern void splt_epilogue_kernel_launch(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t cDataType, int32_t outDataType,
    uint32_t blockDim, void *stream);

// Fused matmul+epilogue kernel for splitK==1
// AIC: matmul -> Fixpipe L0C->UB -> CrossCore notify AIV
// AIV: CrossCore wait -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
// outDataType — INT8 path selects INT8 (saturation Cast) vs INT32 (direct write).
// cDataType — 同 splt_epilogue_kernel_launch。
extern void splt_fused_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm,
    GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t cDataType, int32_t outDataType,
    uint32_t blockDim, void *stream);

#ifdef __cplusplus
}
#endif

// aclsparseLtMatmul 的执行核心（校验链与 kernel launch 顺序与公共入口
// 一致）。仅 host CXX TU 消费（matmul_host.cpp 定义，matmul_search_host.cpp
// 调用）；ASC TU 跳过（aclsparseStatus_t/aclrtStream 为 host-only 类型）。
#if defined(__cplusplus) && !defined(__CCE_AICORE__)
aclsparseStatus_t spltMatmulRunInternal(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams);
#endif // __cplusplus && !__CCE_AICORE__

#endif // SPLT_MATMUL_KERNEL_H
