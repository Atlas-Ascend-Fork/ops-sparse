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

#ifndef OP_KERNEL_DATA_SPSV_VECTOR_OPS_H_
#define OP_KERNEL_DATA_SPSV_VECTOR_OPS_H_

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateValuesSimtCompute(
    __gm__ const float* newValues, __gm__ float* csrValues, __gm__ const PermT* perm, int64_t nnz)
{
    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < nnz; p += static_cast<int64_t>(blockDim.x)) {
        csrValues[p] = newValues[perm[p]];
    }
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateDiagSimtCompute(
    __gm__ const float* newDiagValues, __gm__ float* values, __gm__ const PermT* diagPtr, int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        PermT dp = diagPtr[i];
        if (dp >= 0) {
            values[dp] = newDiagValues[i];
        }
    }
}

// SpsvFillZero / SpsvScaleCopy / SpsvCopyValues: small-vector helper kernels.
// These operate on vectors of length m (matrix dimension), which is typically
// small for triangular solve workloads. SIMT per-element writes are adequate
// here; a DataCopy/Duplicate path would add TPipe/TQue setup overhead that
// outweighs the benefit for these small transfers.
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvFillZeroSimtCompute(__gm__ float* vecY,
                                                                                              int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = 0.0f;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleCopySimtCompute(__gm__ const float* vecX,
                                                                                               __gm__ float* vecY,
                                                                                               float alpha, int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = alpha * vecX[i];
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleInfSimtCompute(__gm__ const float* vecX,
                                                                                              __gm__ float* vecY,
                                                                                              float alpha, int64_t m)
{
    const float inf = 1.0f / 0.0f;
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        vecY[i] = (alpha * vecX[i]) * inf;
    }
}

// complex64 counterparts.  The vectors hold m complex elements, so running the
// float helpers over m elements would write half the buffer and ignore alpha's
// imaginary part.  x is read into a local before vecY[i] is written, which is
// what makes an in-place solve (vecY aliasing vecX) safe: a lane only writes
// the element it has already read.
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleCopyComplexSimtCompute(
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, SpsvComplex64 alpha, int64_t m)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        SpsvComplex64 x = {vecX[i].real, vecX[i].imag};
        SpsvComplex64 scaled = SpsvComplexMul(alpha, x);
        vecY[i].real = scaled.real;
        vecY[i].imag = scaled.imag;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvScaleInfComplexSimtCompute(
    __gm__ const volatile SpsvComplex64* vecX, __gm__ SpsvComplex64* vecY, SpsvComplex64 alpha, int64_t m)
{
    const float inf = 1.0f / 0.0f;
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        SpsvComplex64 x = {vecX[i].real, vecX[i].imag};
        SpsvComplex64 scaled = SpsvComplexMul(alpha, x);
        vecY[i].real = scaled.real * inf;
        vecY[i].imag = scaled.imag * inf;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvCopyValuesSimtCompute(
    __gm__ const float* src, __gm__ float* dst, int64_t nnz, int32_t blockId, int32_t numBlocks)
{
    int64_t globalTid = static_cast<int64_t>(blockId) * blockDim.x + threadIdx.x;
    int64_t globalStride = static_cast<int64_t>(numBlocks) * blockDim.x;
    for (int64_t p = globalTid; p < nnz; p += globalStride) {
        dst[p] = src[p];
    }
}

#endif // OP_KERNEL_DATA_SPSV_VECTOR_OPS_H_
