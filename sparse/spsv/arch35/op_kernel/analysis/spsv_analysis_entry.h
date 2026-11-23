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

#ifndef OP_KERNEL_ANALYSIS_SPSV_ANALYSIS_ENTRY_H_
#define OP_KERNEL_ANALYSIS_SPSV_ANALYSIS_ENTRY_H_

__simt_vf__ __aicore__ __launch_bounds__(kSimtMaxThreads) inline void SpsvAnalysisComplexSimtCompute(
    __gm__ const RowPtrT* rowPtr, __gm__ const ColIndT* colInd, __gm__ const SpsvComplex64* values,
    __gm__ uint8_t* workspace, int64_t m, int64_t nnz, int32_t fillMode, int32_t opA, int32_t format, int32_t numSlices,
    int32_t sliceWidth, int32_t conjValues, int64_t levelPtrOffset, int64_t levelRowOffset, int64_t diagPtrOffset,
    int64_t validCountOffset, int64_t csrRowPtrOffset, int64_t csrColIndOffset, int64_t csrValuesOffset,
    int64_t permOffset, int64_t transRowPtrOffset, int64_t transColIndOffset, int64_t transValuesOffset,
    int64_t transPermOffset, int32_t idxBase, int32_t numBlocks)
{
    // Thread 0 does the structure: the level chain is sequential and the
    // conversion is kept deterministic.  The histogram below is collective,
    // so the block's other threads fall through to it instead of returning.
    if (threadIdx.x == 0) {
        __gm__ RowPtrT* csrRp = (__gm__ RowPtrT*)(workspace + csrRowPtrOffset);
        __gm__ ColIndT* csrCi = (__gm__ ColIndT*)(workspace + csrColIndOffset);
        __gm__ SpsvComplex64* csrVal = (__gm__ SpsvComplex64*)(workspace + csrValuesOffset);
        __gm__ RowPtrT* trRp = (__gm__ RowPtrT*)(workspace + transRowPtrOffset);
        __gm__ ColIndT* trCi = (__gm__ ColIndT*)(workspace + transColIndOffset);
        __gm__ SpsvComplex64* trVal = (__gm__ SpsvComplex64*)(workspace + transValuesOffset);
        __gm__ PermT* perm = (__gm__ PermT*)(workspace + permOffset);
        __gm__ PermT* trPerm = (__gm__ PermT*)(workspace + transPermOffset);
        __gm__ PermT* diagPtr = (__gm__ PermT*)(workspace + diagPtrOffset);
        __gm__ int32_t* levelPtr = (__gm__ int32_t*)(workspace + levelPtrOffset);
        __gm__ int32_t* levelRow = (__gm__ int32_t*)(workspace + levelRowOffset);
        __gm__ int32_t* validCount = (__gm__ int32_t*)(workspace + validCountOffset);

        __gm__ const RowPtrT* srcRp = rowPtr;
        __gm__ const ColIndT* srcCi = colInd;
        __gm__ const SpsvComplex64* srcVal = values;
        bool converted = SpsvComplexNormaliseToCsr<RowPtrT, ColIndT, PermT>(rowPtr, colInd, values, csrRp, csrCi,
                                                                            csrVal, perm, diagPtr, m, nnz, format,
                                                                            numSlices, sliceWidth, conjValues, idxBase);
        if (converted) {
            srcRp = csrRp;
            srcCi = csrCi;
            srcVal = csrVal;
        }

        __gm__ const RowPtrT* effRp = srcRp;
        __gm__ const ColIndT* effCi = srcCi;
        if (opA != 0) {
            SpsvComplexBuildTranspose<RowPtrT, ColIndT, PermT>(srcRp, srcCi, srcVal, trRp, trCi, trVal, trPerm, perm,
                                                               diagPtr, m, nnz, opA, converted);
            effRp = trRp;
            effCi = trCi;
        }

        SpsvBuildComplexAnalysisLevels(effRp, effCi, diagPtr, levelPtr, levelRow, validCount, workspace, m, fillMode,
                                       opA, numBlocks);
    }
    SpsvFinishComplexAnalysisHistogram<PermT>(workspace, diagPtrOffset, levelPtrOffset, m, numBlocks);
}

extern "C" __global__ __aicore__ void spsv_analysis_complex_kernel(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                                   GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // Single block: this is the serial phase.  The structural conversion
    // stays deterministic here, and the levelRow scatter and
    // diagPtr/validCount go to the shared multi-block parallel phase.
    if (tiling.indexType == 0 && tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            asc_vf_call<SpsvAnalysisComplexSimtCompute<int32_t, int32_t, int32_t>>(
                dim3{tiling.nthreads, 1, 1}, (__gm__ const int32_t*)rowPtr, (__gm__ const int32_t*)colInd,
                (__gm__ const SpsvComplex64*)values, (__gm__ uint8_t*)workspace, tiling.m, tiling.nnz, tiling.fillMode,
                tiling.opA, tiling.format, tiling.numSlices, tiling.sliceWidth, tiling.conjValues,
                tiling.levelPtrOffset, tiling.levelRowOffset, tiling.diagPtrOffset, tiling.validCountOffset,
                tiling.csrRowPtrOffset, tiling.csrColIndOffset, tiling.csrValuesOffset, tiling.permOffset,
                tiling.transRowPtrOffset, tiling.transColIndOffset, tiling.transValuesOffset, tiling.transPermOffset,
                tiling.idxBase, tiling.numBlocks);
        }
    }
}

#define SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, FWD, NU, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1, e2, nt) \
    asc_vf_call<Func<RowPtrT, ColIndT, FWD, NU, PermT>>(                                                            \
        dim3{nt, 1, 1}, (__gm__ const RowPtrT*)rp, (__gm__ const ColIndT*)ci, (__gm__ const float*)val,             \
        (__gm__ const float*)vx, (__gm__ float*)vy, yLow, lp, lr, dp, vc, a, e1, e2)

#define SPSV_CALL_SOLVE_COMPLEX(RowPtrT, ColIndT, PermT, FWD, NU, SR, rp, ci, val, vx, vy, yw, lp, lr, dp, vc, a, e1, \
                                e2, nt)                                                                               \
    asc_vf_call<SpsvSolveComplexSimtCompute<RowPtrT, ColIndT, FWD, NU, PermT, SR>>(                                   \
        dim3{nt, 1, 1}, (__gm__ const RowPtrT*)rp, (__gm__ const ColIndT*)ci, (__gm__ const SpsvComplex64*)val,       \
        (__gm__ const SpsvComplex64*)vx, (__gm__ SpsvComplex64*)vy, yw, lp, lr, dp, vc, a, 0, e2)

#define SPSV_DISPATCH_SOLVE(RowPtrT, ColIndT, PermT, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1, e2, nt, fwd,    \
                            nonUnit)                                                                                   \
    do {                                                                                                               \
        if (fwd && nonUnit) {                                                                                          \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, true, true, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1, e2, \
                            nt);                                                                                       \
        } else if (fwd) {                                                                                              \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, true, false, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1,    \
                            e2, nt);                                                                                   \
        } else if (nonUnit) {                                                                                          \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, false, true, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1,    \
                            e2, nt);                                                                                   \
        } else {                                                                                                       \
            SPSV_CALL_SOLVE(RowPtrT, ColIndT, PermT, false, false, Func, rp, ci, val, vx, vy, lp, lr, dp, vc, a, e1,   \
                            e2, nt);                                                                                   \
        }                                                                                                              \
    } while (0)

// Single-core analysis path: one block runs the whole analysis (format
// conversion, level computation, scatter, diag/validCount, finalize) using
// block-level asc_syncthreads. The multi-core path is split into three
// separate host-side kernel launches (serial / parallel / final) below.
extern "C" __global__ __aicore__ void spsv_analysis_kernel(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                           GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Dispatch on (indexType, colIndType, permType):
    //   I32 ptr + I32 col + permType=0: PermT int32.
    //   I32 ptr + I32 col + permType=1: PermT int64 (SPSV_FORCE_PERMT_64=1).
    //   I64 ptr + I32 col + permType=0: PermT int32.
    //   I64 ptr + I32 col + permType=1: PermT int64 (nnz > INT32_MAX).
    //   I64 ptr + I64 col + permType=0: PermT int32.
    //   I64 ptr + I64 col + permType=1: PermT int64.
    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int32_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int64_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int32_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int64_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int32_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int64_t, SpsvAnalysisSimtCompute, rowPtr, colInd, values, workspace,
                               tiling);
        }
    }
}

// Multi-core analysis, phase 1 (serial): single-block launch. Performs format
// conversion, sequential level computation, histogram and prefix-sum. Stream
// ordering relative to the next launch provides cross-kernel GM visibility,
// replacing the in-kernel SyncAll the old single-launch design relied on.
extern "C" __global__ __aicore__ void spsv_analysis_serial_kernel(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values,
                                                                  GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int32_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int32_t, int32_t, int64_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int32_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int32_t, int64_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int32_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        } else {
            SPSV_CALL_ANALYSIS(int64_t, int64_t, int64_t, SpsvAnalysisSerialPhase, rowPtr, colInd, values, workspace,
                               tiling);
        }
    }
}

// Multi-core analysis, phase 2 (parallel): launched with numBlocks blocks.
// Scatter (levelRow fill), diagonal-index and validCount computation. Each
// block works on a disjoint row partition with no cross-block data dependency,
// so no barrier is needed within this kernel.
extern "C" __global__ __aicore__ void spsv_analysis_parallel_kernel(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace,
                                                                    const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.indexType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int32_t, int32_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int32_t, int32_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    } else if (tiling.colIndType == 0) {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int32_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int32_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    } else {
        if (tiling.permType == 0) {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int64_t, int32_t, rowPtr, colInd, workspace, tiling);
        } else {
            SPSV_CALL_ANALYSIS_PAR(int64_t, int64_t, int64_t, rowPtr, colInd, workspace, tiling);
        }
    }
}

// Multi-core analysis, phase 3 (final): single-block launch. Restores the
// levelPtr prefix-sum (right-shift) and writes numLevels.
extern "C" __global__ __aicore__ void spsv_analysis_final_kernel(GM_ADDR workspace, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<SpsvAnalysisFinalPhase>(dim3{tiling.nthreads, 1, 1}, (__gm__ uint8_t*)workspace, tiling.levelPtrOffset);
}

#endif // OP_KERNEL_ANALYSIS_SPSV_ANALYSIS_ENTRY_H_
