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
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "spsv_kernel.h"
#include "math/spsv_arith.h"
#include "math/spsv_range.h"
#include "math/spsv_exact.h"
#include "recovery/spsv_rational.h"
#include "recovery/spsv_dyadic.h"
#include "validation/spsv_block_bound.h"
#include "validation/spsv_two_band_bound.h"
#include "validation/spsv_inverse_bound.h"
#include "analysis/spsv_analysis_impl.h"

#include "common/spsv_schedule_common.h"
#include "solve/spsv_complex_row_math.h"
#include "common/spsv_dispatch_types.h"
#include "validation/spsv_exact_row.h"
#include "validation/spsv_enclosed_arithmetic.h"
#include "validation/spsv_complex_row.h"
#include "validation/spsv_complex_preparation.h"
#include "solve/spsv_complex_fast_row.h"
#include "solve/spsv_complex_levels.h"
#include "solve/spsv_real_shadow.h"
#include "solve/spsv_real_schedule.h"
#include "common/spsv_vector_ops.h"
#include "analysis/spsv_complex_analysis.h"
#include "analysis/spsv_analysis_entry.h"
#include "dispatch/spsv_real_dispatch.h"
#include "validation/spsv_real_dependency.h"
#include "validation/spsv_dependency_recovery.h"
#include "dispatch/spsv_schedule_dispatch.h"
#include "dispatch/spsv_solve_entry.h"

#include "common/spsv_primary_status.h"

extern "C" __schedmode__(1) __global__ __aicore__ void spsv_solve_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvRealPrimaryCertified(workspace, tiling)) {
        return;
    }

    SpsvSolveFloatEntry(rowPtr, colInd, values, vecX, vecY, workspace, tiling);
    int32_t completedLevels=((__gm__ const SpsvTilingData *)workspace)->numLevels;
    // The following signed residual tree replaces the independent inverse
    // attempt for a full deep chain; its scratch summaries are invalidated.
    if (completedLevels > 1 && !(completedLevels == tiling.m && tiling.m >= 128 && tiling.nnz <= 12 * tiling.m)) {
        if (tiling.numBlocks > 1) { SpsvSyncSolveBlocks(); }
        float alpha = tiling.alphaDevicePtr != 0 ? *(__gm__ const float *)tiling.alphaDevicePtr : tiling.alpha;
        asc_vf_call<SpsvFloatInverseCertificate>(dim3{128, 1, 1},
            (__gm__ const uint8_t *)rowPtr, (__gm__ const uint8_t *)colInd,
            (__gm__ const float *)values, (__gm__ const float *)vecX,
            (__gm__ const float *)vecY, (__gm__ float *)(workspace + tiling.yWideOffset) + tiling.m,
            (__gm__ const uint8_t *)(workspace + tiling.diagPtrOffset), alpha, tiling.m,
            (tiling.fillMode == 0) == (tiling.opA == 0), tiling.diagType == 0,
            tiling.indexType != 0, tiling.colIndType != 0, tiling.permType != 0);
    }
}

// Separate resource allocation: adding an FP32 error certificate must not
// raise the register/spill requirements of the ordinary complex64 solve.
extern "C" __schedmode__(1) __global__ __aicore__ void spsv_solve_complex_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvUseEarlyComplexBinary64Primary(tiling) && SpsvBinary64PrimaryCompleted(workspace, tiling.m)) {
        return;
    }
    if (tiling.m >= 1024 && AscendC::GetBlockIdx() == 0) {
        ((__gm__ SpsvTilingData*)workspace)->primaryStatus = 0;
        AscendC::GlobalTensor<uint8_t> state;
        state.SetGlobalBuffer(workspace);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(state);
    }
    SpsvSolveComplexEntry(rowPtr, colInd, values, vecX, vecY, workspace, tiling);
}

extern "C" __global__ __aicore__ void spsv_dependency_retry_kernel(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (SpsvRealPrimaryCertified(workspace, tiling)) {
        return;
    }
    __gm__ const int32_t *levelPtr;
    __gm__ const int32_t *levelRow;
    __gm__ const int32_t *validCount;
    int32_t numLevels;
    bool forward, nonUnit;
    float alpha;
    PrepareSolveKernelArgs(tiling, (__gm__ uint8_t *)workspace,
        levelPtr, levelRow, validCount, numLevels, forward, nonUnit, alpha);
    if (numLevels <= 1) { return; }
    __gm__ float *state = (__gm__ float *)(workspace + tiling.yWideOffset) + tiling.m;
    asc_vf_call<SpsvFloatDependencyRetry>(dim3{256, 1, 1},
        (__gm__ const uint8_t *)rowPtr, (__gm__ const uint8_t *)colInd,
        (__gm__ const float *)values, (__gm__ const float *)vecX,
        (__gm__ float *)vecY, state,
        (__gm__ const uint8_t *)(workspace + tiling.diagPtrOffset),
        alpha, tiling.m, forward, nonUnit, tiling.indexType != 0,
        tiling.colIndType != 0, tiling.permType != 0, static_cast<int32_t>(tiling.numBlocks));
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateValuesComplexSimtCompute(
    __gm__ const SpsvComplex64 *newValues, __gm__ SpsvComplex64 *dst,
    __gm__ const PermT *perm, int64_t nnz, int32_t opA)
{
    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < nnz; p += static_cast<int64_t>(blockDim.x)) {
        PermT q = perm[p];
        dst[p].real = newValues[q].real;
        dst[p].imag = (opA == 2) ? -newValues[q].imag : newValues[q].imag;
    }
}

template <typename PermT>
__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvUpdateDiagComplexSimtCompute(
    __gm__ const SpsvComplex64 *newDiag, __gm__ SpsvComplex64 *values,
    __gm__ const PermT *diagPtr, int64_t m, int32_t opA)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < m; i += static_cast<int64_t>(blockDim.x)) {
        PermT dp = diagPtr[i];
        if (dp >= 0) {
            values[dp].real = newDiag[i].real;
            values[dp].imag = (opA == 2) ? -newDiag[i].imag : newDiag[i].imag;
        }
    }
}

// The complex update passes dispatch on the width the permutation was stored
// at, which is only known at runtime, so every call site needs both
// instantiations.  One macro, the way the solve side already does it, rather
// than three more pairs of 250-column lines.
#define SPSV_CALL_COMPLEX_UPDATE(FN, DST, OFFSET, COUNT)          \
    do {                                                          \
        if (tiling.permType == 0) {                               \
            asc_vf_call<FN<int32_t>>(dim3{tiling.nthreads, 1, 1}, \
                (__gm__ const SpsvComplex64 *)newValues, (DST),   \
                (__gm__ const int32_t *)(workspace + (OFFSET)),   \
                (COUNT), tiling.opA);                             \
        } else {                                                  \
            asc_vf_call<FN<int64_t>>(dim3{tiling.nthreads, 1, 1}, \
                (__gm__ const SpsvComplex64 *)newValues, (DST),   \
                (__gm__ const int64_t *)(workspace + (OFFSET)),   \
                (COUNT), tiling.opA);                             \
        }                                                         \
    } while (0)

extern "C" __global__ __aicore__ void spsv_update_values_kernel(
    GM_ADDR newValues, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.valueType == 1) {
        __gm__ SpsvComplex64 *dst = (__gm__ SpsvComplex64 *)(workspace + tiling.csrValuesOffset);
        SPSV_CALL_COMPLEX_UPDATE(SpsvUpdateValuesComplexSimtCompute, dst,
                                 tiling.permOffset, tiling.nnz);
        return;
    }
    __gm__ float *csrValues = (__gm__ float *)(workspace + tiling.csrValuesOffset);

    if (tiling.permType == 0) {
        __gm__ const int32_t *perm = (__gm__ const int32_t *)(workspace + tiling.permOffset);
        asc_vf_call<SpsvUpdateValuesSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, perm, tiling.nnz);
    } else {
        __gm__ const int64_t *perm = (__gm__ const int64_t *)(workspace + tiling.permOffset);
        asc_vf_call<SpsvUpdateValuesSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, perm, tiling.nnz);
    }
}

extern "C" __global__ __aicore__ void spsv_update_diag_kernel(
    GM_ADDR newValues, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.valueType == 1) {
        __gm__ SpsvComplex64 *dst = (__gm__ SpsvComplex64 *)(workspace + tiling.csrValuesOffset);
        SPSV_CALL_COMPLEX_UPDATE(SpsvUpdateDiagComplexSimtCompute, dst,
                                 tiling.diagPtrOffset, tiling.m);
        return;
    }
    __gm__ float *csrValues = (__gm__ float *)(workspace + tiling.csrValuesOffset);

    if (tiling.permType == 0) {
        __gm__ const int32_t *diagPtr = (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, diagPtr, tiling.m);
    } else {
        __gm__ const int64_t *diagPtr = (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, csrValues, diagPtr, tiling.m);
    }
}

extern "C" __global__ __aicore__ void spsv_update_diag_csr_kernel(
    GM_ADDR newValues, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.valueType == 1) {
        SPSV_CALL_COMPLEX_UPDATE(SpsvUpdateDiagComplexSimtCompute,
                                 (__gm__ SpsvComplex64 *)values,
                                 tiling.diagPtrOffset, tiling.m);
        return;
    }

    if (tiling.permType == 0) {
        __gm__ const int32_t *diagPtr = (__gm__ const int32_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int32_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, (__gm__ float *)values, diagPtr, tiling.m);
    } else {
        __gm__ const int64_t *diagPtr = (__gm__ const int64_t *)(workspace + tiling.diagPtrOffset);
        asc_vf_call<SpsvUpdateDiagSimtCompute<int64_t>>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const float *)newValues, (__gm__ float *)values, diagPtr, tiling.m);
    }
}

extern "C" __global__ __aicore__ void spsv_fill_zero_kernel(
    GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<SpsvFillZeroSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ float *)vecY, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_scale_copy_kernel(
    GM_ADDR vecX, GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.valueType == 1) {
        SpsvComplex64 alphaComplex = ReadComplexAlpha(tiling);
        asc_vf_call<SpsvScaleCopyComplexSimtCompute>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const SpsvComplex64 *)vecX, (__gm__ SpsvComplex64 *)vecY,
            alphaComplex, tiling.m);
        return;
    }

    float alpha = tiling.alpha;
    if (tiling.alphaDevicePtr != 0) {
        alpha = *(__gm__ const float *)(tiling.alphaDevicePtr);
    }

    asc_vf_call<SpsvScaleCopySimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)vecX, (__gm__ float *)vecY, alpha, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_scale_inf_kernel(
    GM_ADDR vecX, GM_ADDR vecY, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.valueType == 1) {
        SpsvComplex64 alphaComplex = ReadComplexAlpha(tiling);
        asc_vf_call<SpsvScaleInfComplexSimtCompute>(dim3{tiling.nthreads, 1, 1},
            (__gm__ const SpsvComplex64 *)vecX, (__gm__ SpsvComplex64 *)vecY,
            alphaComplex, tiling.m);
        return;
    }

    float alpha = tiling.alpha;
    if (tiling.alphaDevicePtr != 0) {
        alpha = *(__gm__ const float *)(tiling.alphaDevicePtr);
    }

    asc_vf_call<SpsvScaleInfSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)vecX, (__gm__ float *)vecY, alpha, tiling.m);
}

extern "C" __global__ __aicore__ void spsv_copy_values_kernel(
    GM_ADDR src, GM_ADDR dst, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockId = static_cast<int32_t>(AscendC::GetBlockIdx());
    asc_vf_call<SpsvCopyValuesSimtCompute>(dim3{tiling.nthreads, 1, 1},
        (__gm__ const float *)src, (__gm__ float *)dst, tiling.nnz, blockId,
        static_cast<int32_t>(tiling.numBlocks));
}

// ---------------------------------------------------------------------------
// Host-side kernel launchers
// These functions are compiled with host toolchain and invoke the device
// kernels via the <<<>>> launch syntax. Kept in this file per repo convention
// (spmm/gpsv follow the same pattern).
// ---------------------------------------------------------------------------

void spsv_analysis_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_analysis_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtr, colInd, values, workspace, tiling);
}

void spsv_analysis_complex_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData &tiling, void *stream)
{
    spsv_analysis_complex_kernel<<<1, nullptr, stream>>>(
        rowPtr, colInd, values, workspace, tiling);
}

void spsv_analysis_serial_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
    const SpsvTilingData &tiling, void *stream)
{
    // One AIV block runs serial analysis on SIMT thread 0.
    spsv_analysis_serial_kernel<<<1, nullptr, stream>>>(
        rowPtr, colInd, values, workspace, tiling);
}

void spsv_analysis_parallel_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_analysis_parallel_kernel<<<numBlocks, nullptr, stream>>>(
        rowPtr, colInd, workspace, tiling);
}

void spsv_analysis_final_kernel_do(
    GM_ADDR workspace, const SpsvTilingData &tiling, void *stream)
{
    // Finalize the schedule on SIMT thread 0 of one AIV block.
    spsv_analysis_final_kernel<<<1, nullptr, stream>>>(workspace, tiling);
}

void spsv_float_prefix_certificate_kernel_do(GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,const SpsvTilingData&,void*);

void spsv_solve_kernel_do(
    GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
    GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    if (SpsvUseRealScaledPrimary(tiling)) {
        spsv_real_scaled_primary_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling, stream);
    }
    if (SpsvUseRealBinary64Primary(tiling)) {
        spsv_real_binary64_primary_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling, stream);
    }
    bool earlyComplex = SpsvUseEarlyComplexBinary64Primary(tiling);
    if (earlyComplex) {
        spsv_complex_binary64_primary_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling, stream, true);
    }
    if (tiling.valueType == 1) {
        spsv_solve_complex_kernel<<<numBlocks, nullptr, stream>>>(
            rowPtr, colInd, values, vecX, vecY, workspace, tiling);
    } else {
        spsv_solve_kernel<<<numBlocks, nullptr, stream>>>(
            rowPtr, colInd, values, vecX, vecY, workspace, tiling);
    }
    if (SpsvUseComplexBinary64Primary(tiling) && !earlyComplex) {
        spsv_complex_binary64_primary_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling, stream);
    }
    if (tiling.valueType == 1) {
        if (tiling.opA != 0 && tiling.transValuesOffset >= 0) {
            rowPtr = workspace + tiling.transRowPtrOffset;
            colInd = workspace + tiling.transColIndOffset;
            values = workspace + tiling.transValuesOffset;
        }
        // Only a completed snapshot of this analysis may suppress the launches.
        // Unknown/pending schedules retain the original dependency recovery.
        if (tiling.m > 1 && tiling.numLevels != 1) {
            // A pending snapshot cannot prove that recovery is unnecessary.
            spsv_complex_dyadic_recovery_kernel_do(
                rowPtr, colInd, values, vecX, vecY, workspace, tiling, stream);
        }
    } else if (tiling.m > 1 && tiling.numLevels != 1) {
        // Match the recovery kernel's single-level early return. Pending
        // schedules (numLevels == 0) still run dependency recovery.
        if (tiling.m >= 128) {
            spsv_float_prefix_certificate_kernel_do(rowPtr,colInd,values,vecX,vecY,workspace,tiling,stream);
        }
        spsv_dependency_retry_kernel<<<1, nullptr, stream>>>(
            rowPtr, colInd, values, vecX, vecY, workspace, tiling);
    }
}

void spsv_update_values_kernel_do(
    GM_ADDR newValues, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_values_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, workspace, tiling);
}

void spsv_update_diag_kernel_do(
    GM_ADDR newValues, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_diag_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, workspace, tiling);
}

void spsv_update_diag_csr_kernel_do(
    GM_ADDR newValues, GM_ADDR values, GM_ADDR workspace,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_update_diag_csr_kernel<<<numBlocks, nullptr, stream>>>(
        newValues, values, workspace, tiling);
}

void spsv_fill_zero_kernel_do(
    GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_fill_zero_kernel<<<numBlocks, nullptr, stream>>>(vecY, tiling);
}

void spsv_scale_copy_kernel_do(
    GM_ADDR vecX, GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_scale_copy_kernel<<<numBlocks, nullptr, stream>>>(vecX, vecY, tiling);
}

void spsv_scale_inf_kernel_do(
    GM_ADDR vecX, GM_ADDR vecY,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_scale_inf_kernel<<<numBlocks, nullptr, stream>>>(vecX, vecY, tiling);
}

void spsv_copy_values_kernel_do(
    GM_ADDR src, GM_ADDR dst,
    uint32_t numBlocks, const SpsvTilingData &tiling, void *stream)
{
    spsv_copy_values_kernel<<<numBlocks, nullptr, stream>>>(src, dst, tiling);
}
