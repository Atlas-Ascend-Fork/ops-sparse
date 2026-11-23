/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_SPMM_1D_PLAN_INTERNAL_H_
#define CUBE_SPMM_1D_PLAN_INTERNAL_H_
#include "cann_ops_sparse.h"
/** Immutable execution plan. Host row offsets are analyzed once at creation;
 * execution requires the same sparse structure on the device. Data values and
 * B may change. Plan creation/analysis is outside the device-task timing. */
struct CubeSpmm1dInternalPlan;
typedef struct CubeSpmm1dInternalPlan* CubeSpmm1dInternalPlan_t;

typedef enum CubeSpmm1dInternalPolicy {
    CUBE_SPMM_1D_INTERNAL_POLICY_PHASE1 = 1,
    CUBE_SPMM_1D_INTERNAL_POLICY_EXPLICIT = 2,
    CUBE_SPMM_1D_INTERNAL_POLICY_STRUCTURAL = 3
} CubeSpmm1dInternalPolicy;

enum {
    CUBE_SPMM_1D_INTERNAL_PLAN_TILE_PADDING = 1,
    CUBE_SPMM_1D_INTERNAL_PLAN_RULE_PADDING = 4,
    /* Caller guarantees finite A/B values. Enables block-diagonal row fusion;
     * generic plans retain the original nonfinite-value behavior. */
    CUBE_SPMM_1D_INTERNAL_PLAN_FINITE_INPUTS = 8
};

typedef struct CubeSpmm1dInternalPlanOptions {
    size_t structSize;
    CubeSpmm1dInternalPolicy policy;
    int32_t tilingKey;          /* Used only for EXPLICIT. */
    uint32_t flags;             /* CUBE_SPMM_1D_INTERNAL_PLAN_* flags. */
    uint64_t allowedKeys;       /* Bit key; zero permits every compiled key. */
    size_t workspaceLimitBytes; /* Zero means no additional caller limit. */
} CubeSpmm1dInternalPlanOptions;

typedef struct CubeSpmm1dInternalPlanInfo {
    int32_t m, originalN, paddedN, k, vectorLength;
    int32_t deviceId, runtimeCores, blockDim, tilingKey;
    int32_t aicNBatch, sparseKTile, bPackColumns;
    int64_t maxVectorsPerRow, totalVectors, emptyRowBlocks;
    size_t workspaceBytes;
    int32_t policyApplied;
} CubeSpmm1dInternalPlanInfo;

/* Host offsets must start at zero, be monotone, and contain m/16+1 entries.
 * options=NULL uses STRUCTURAL and RULE_PADDING. FINITE_INPUTS enables row
 * fusion on the calibrated 20-core 910B3. Masks and workspace budgets reject
 * unsupported selections. The plan is reusable with the same sparse structure. */
aclsparseStatus_t CubeSpmm1dInternalCreatePlan(aclsparseHandle_t handle, int32_t m, int32_t n, int32_t k,
                                               const int32_t* hostRowOffsets, size_t offsetCount,
                                               const CubeSpmm1dInternalPlanOptions* options,
                                               CubeSpmm1dInternalPlan_t* plan);
aclsparseStatus_t CubeSpmm1dInternalGetPlanInfo(CubeSpmm1dInternalPlan_t plan, CubeSpmm1dInternalPlanInfo* info);
aclsparseStatus_t CubeSpmm1dInternalDestroyPlan(CubeSpmm1dInternalPlan_t plan);

/* B uses plan.paddedN as its row stride. C is FP32 column-major and follows
 * the original caller-preclear contract: initialize C to zero before execution
 * when empty row blocks may exist. The kernel skips empty row blocks. Reusing
 * C with the same sparse structure keeps those zero entries valid. Clearing C
 * is outside the MIX_AIC device-task timing, as in the original implementation. */
aclsparseStatus_t CubeSpmm1dInternalExecute(aclsparseHandle_t handle, CubeSpmm1dInternalPlan_t plan, const void* values,
                                            const int32_t* columnIndices, const int32_t* rowOffsets, const void* b,
                                            void* c, void* workspace, size_t workspaceBytes);

#endif
