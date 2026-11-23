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

#include "recovery/spsv_complex_recovery_common.h"
#include "common/spsv_complex_workspace.h"
#include "validation/spsv_complex_certificate_pipeline.h"
#include "recovery/spsv_dyadic.h"
#include "recovery/spsv_complex_dyadic_core.h"
#include "recovery/spsv_complex_dyadic_division.h"
#include "validation/spsv_complex_prefix.h"
#include "validation/spsv_complex_triple_prefix.h"
#include "validation/spsv_complex_border.h"
#include "aclsparse_host_utils.h"
#include "validation/spsv_complex_certificate_schedule.h"
#include "validation/spsv_complex_head_certificate.h"


__simt_callee__ inline void SpsvClassifyExactComplexRows(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m,
    bool forward, bool nonUnit, __gm__ int32_t* scale, __gm__ float* rowPrep)
{
    for (int64_t step = 0; step < m; ++step) {
        int64_t r = forward ? step : m - 1 - step;
        SpsvComplex64 diagonal{1.0f, 0.0f};
        if (nonUnit) {
            diagonal = diagPtr[r] >= 0 ? SpsvComplex64{values[diagPtr[r]].real, values[diagPtr[r]].imag}
                                       : SpsvComplex64{0.0f, 0.0f};
        }
        bool valid = SpsvIsFiniteF(alpha.real) && SpsvIsFiniteF(alpha.imag) && SpsvIsFiniteF(rhs[r].real) &&
                     SpsvIsFiniteF(rhs[r].imag) && SpsvIsFiniteF(diagonal.real) &&
                     SpsvIsFiniteF(diagonal.imag) &&
                     (SpsvRationalNonzero(diagonal.real) || SpsvRationalNonzero(diagonal.imag));
        for (int64_t p = rowPtr[r]; p < rowPtr[r + 1]; ++p) {
            int32_t c = colInd[p];
            if (forward ? c < r : c > r) {
                valid &= SpsvIsFiniteF(values[p].real) && SpsvIsFiniteF(values[p].imag) && scale[c] != 0;
            }
        }
        scale[r] = valid ? 1 : 0;
        if (!valid) {
            rowPrep[18 * r + 9] = rowPrep[18 * r + 10] = kSpsvInvalidDependencyRadius;
        }
    }
}

// The fast complex pass publishes componentwise forward-error enclosures
// in the consumed RHS-bound slots. Only uncertified dependency subgraphs
// enter exact reconstruction, against the immutable original RHS.
__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvComplexDyadicRecovery(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* vecY, __gm__ float* yWide,
    __gm__ const int32_t* diagPtr, SpsvComplex64 alpha, int64_t m, bool forward, bool nonUnit,
    int32_t riskBlocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)yWide);
    }
    asc_syncthreads();
    SpsvComplexScratchView scratch{yWide, {static_cast<uint64_t>(m)}};
    __gm__ float* rowPrep = scratch.rowPrep;
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    uint32_t risk = 0;
    __gm__ const uint32_t* blockRisk = scratch.risk;
    for (int32_t block = tid; block < riskBlocks; block += blockDim.x) {
        risk |= blockRisk[block];
    }
    risk = asc_reduce_max(risk);
    if (tid != 0) {
        return;
    }
    __gm__ int32_t* scale = scratch.scale;
    // Clear the continuation token on every solve because this workspace is
    // reused by descriptor updates and in-place calls.
    *scratch.continuation = 0;
    asc_threadfence();
    if (risk == 0) {
        return;
    }
    __gm__ uint16_t* digits = scratch.digits;
    SpsvClassifyExactComplexRows(rowPtr, colInd, values, rhs, diagPtr, alpha, m, forward, nonUnit, scale,
                                 rowPrep);
    bool remaining = SpsvComplexDyadicRetry(rowPtr, colInd, values, rhs, diagPtr, alpha, m,
                                                          forward, nonUnit, vecY, yWide, rowPrep, digits);
    // The classifier and dyadic arena consume the original risk vector.
    // Publish an explicit continuation token for the next same-stream kernel;
    // no dyadic exponent or validity word is interpreted as a certificate.
    *scratch.continuation = remaining ? 1 : 0;
    asc_threadfence();
}

// Keep the rare dependency reconstruction out of the scheduled solve's
// resource footprint. Stream ordering publishes every producer block before
// this single-block consumer starts, including on repeated workspace reuse.
extern "C" __schedmode__(1) __global__ __aicore__
    void spsv_complex_dyadic_recovery_kernel(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY,
                                             GM_ADDR yWide, GM_ADDR diagPtr, GM_ADDR workspace, int64_t m,
                                             int32_t forward, int32_t nonUnit, float alphaReal, float alphaImag,
                                             uint64_t alphaDevicePtr, uint32_t riskBlocks, int32_t fusedPrimary,
                                             int64_t levelPtrOffset, int64_t levelRowOffset, int32_t fusedRational)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvComplexPrimaryCompleted(workspace, m)) {
        return;
    }
    if (((__gm__ const SpsvTilingData*)workspace)->numLevels <= 1) {
        return;
    }
    SpsvComplex64 alpha = SpsvRecoveryAlpha(alphaReal, alphaImag, alphaDevicePtr);
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    auto data = SpsvMakeComplexCertificateData(rowPtr, colInd, values, vecX, vecY, yWide, diagPtr, alpha,
                                               forward, nonUnit);
    if (!SpsvPrepareComplexRecovery(data, m, riskBlocks, block, blocks, fusedPrimary != 0)) {
        return;
    }
    uint32_t scanRiskBlocks = riskBlocks;
    SpsvComplexCertificatePipeline pipeline{data, m, scanRiskBlocks, block, blocks};
    scanRiskBlocks = pipeline.Run<kSpsvTriplePrefixWidth, false>(
        ((__gm__ const SpsvTilingData*)workspace)->numLevels,
        (__gm__ const int32_t*)(workspace + levelPtrOffset), (__gm__ const int32_t*)(workspace + levelRowOffset));
    if (block != 0) {
        return;
    }
    asc_vf_call<SpsvComplexDyadicRecovery>(dim3{32, 1, 1}, (__gm__ const int32_t*)rowPtr,
                                           (__gm__ const int32_t*)colInd, (__gm__ const SpsvComplex64*)values,
                                           (__gm__ const SpsvComplex64*)vecX, (__gm__ SpsvComplex64*)vecY,
                                           (__gm__ float*)yWide, (__gm__ const int32_t*)diagPtr, alpha, m,
                                           forward != 0, nonUnit != 0, static_cast<int32_t>(scanRiskBlocks));
    if (fusedRational != 0) {
        asc_vf_call<SpsvComplexRationalRecoveryBody>(
            dim3{32, 1, 1}, (__gm__ const int32_t*)rowPtr, (__gm__ const int32_t*)colInd,
            (__gm__ const SpsvComplex64*)values, (__gm__ const SpsvComplex64*)vecX, (__gm__ SpsvComplex64*)vecY,
            (__gm__ float*)yWide, (__gm__ const int32_t*)diagPtr, alpha, m, forward != 0, nonUnit != 0,
            static_cast<int32_t>(scanRiskBlocks));
    }
}

void spsv_complex_wide_primary_do(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const SpsvTilingData&,
                                  uint32_t, uint32_t, void*);
void spsv_complex_dyadic_recovery_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                            GM_ADDR vecY, GM_ADDR workspace, const SpsvTilingData& tiling,
                                            void* stream)
{
    uint32_t recoveryBlocks = 1U;
    if (tiling.m >= 128) {
        uint32_t available = GetAivCoreCount();
        int64_t width = tiling.diagType == 0 ? kSpsvTriplePrefixWidth : kSpsvPrefixWidth;
        uint32_t parts = static_cast<uint32_t>((tiling.m + width - 1) / width);
        recoveryBlocks = available > 0U ? (available < parts ? available : parts) : tiling.numBlocks;
    }
    // A shallow, wide schedule already has enough independent row workers.
    // Reuse that budget rather than making every physical core participate
    // in the recovery scan and its two grid rendezvous. Deep and pending
    // schedules retain the full certificate parallelism.
    bool wideLevels = tiling.numLevels > 1 && tiling.numLevels <= 64 && tiling.m / tiling.numLevels >= 32;
    if (wideLevels && tiling.numBlocks > 0U && tiling.numBlocks < recoveryBlocks) {
        recoveryBlocks = tiling.numBlocks;
    }
    // Ordinary matrices share one recovery launch. Tiny known schedules keep
    // the separately compiled CRT entry; pending analysis must retain an exact
    // continuation in the current launch until its actual schedule is known.
    bool fuseRational = tiling.m >= 128 || tiling.numLevels == 0;
    uint32_t riskBlocks = tiling.numBlocks > recoveryBlocks ? tiling.numBlocks : recoveryBlocks;
    bool widePrimary =
        tiling.m >= 1024 && (tiling.nnz / tiling.m >= 13 ||
                             (tiling.diagType == 0 && tiling.m <= 4096 && tiling.nnz / tiling.m >= 10));
    if (widePrimary) {
        spsv_complex_wide_primary_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling, recoveryBlocks,
                                     riskBlocks, stream);
    }
    spsv_complex_dyadic_recovery_kernel<<<recoveryBlocks, nullptr, stream>>>(
        rowPtr, colInd, values, vecX, vecY, workspace + tiling.yWideOffset, workspace + tiling.diagPtrOffset, workspace,
        tiling.m, static_cast<int32_t>((tiling.fillMode == 0) == (tiling.opA == 0)),
        static_cast<int32_t>(tiling.diagType == 0), tiling.alpha, tiling.alphaImag, tiling.alphaDevicePtr, riskBlocks,
        static_cast<int32_t>(!widePrimary), tiling.levelPtrOffset, tiling.levelRowOffset,
        static_cast<int32_t>(fuseRational));
    if (!fuseRational) {
        spsv_complex_rational_recovery_kernel_do(rowPtr, colInd, values, vecX, vecY, workspace, tiling,
                                                 stream);
    }
}
