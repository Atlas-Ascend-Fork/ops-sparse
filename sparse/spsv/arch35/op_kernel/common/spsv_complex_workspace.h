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

#ifndef SPSV_COMPLEX_WORKSPACE_H_
#define SPSV_COMPLEX_WORKSPACE_H_

#include "../../shared/spsv_workspace_layout.h"

// Phase ownership of the aliased arena:
// 1. Scan/certificates publish risk flags; all participating blocks consume them.
// 2. Only the single recovery block remains: classification/dyadic reuse risk as
//    row scales and residues, then publish continuation in scale[0].
// 3. CRT reads continuation before reusing scales/residues for marked systems.
// The existing barriers/fences establish transitions; this view adds none.
struct SpsvComplexScratchView {
    __gm__ float* base;
    SpsvComplexWorkspaceGeometry geometry;
    __gm__ float* rowPrep{base + geometry.rowPrepWord};
    __gm__ float* arena{base + geometry.riskWord};
    __gm__ uint32_t* risk{(__gm__ uint32_t*)arena};
    __gm__ int32_t* scale{(__gm__ int32_t*)risk};
    __gm__ uint32_t* continuation{risk};
    __gm__ uint32_t* residue{(__gm__ uint32_t*)(scale + geometry.rows)};
    __gm__ uint16_t* digits{(__gm__ uint16_t*)(base + geometry.digitWord)};
};

// Certificate stages preserve risk slots, publish one flag per block, then
// reuse the remaining arena. Tree and border layouts share this boundary.
struct SpsvCertificateScratchView {
    SpsvComplexScratchView workspace;
    uint32_t slots;
    uint32_t blocks;
    __gm__ uint32_t* flags{workspace.risk + slots};
    __gm__ float* storage{(__gm__ float*)(flags + blocks)};
    uint64_t storageWord{workspace.geometry.riskWord + slots + blocks};
};

__aicore__ inline bool SpsvCertificateArenaFits(int64_t m, uint64_t required)
{
    return required <= SpsvComplexWorkspaceGeometry{static_cast<uint64_t>(m)}.arenaWords;
}

// Binary64 replay owns this arena only after certificate flags have been
// consumed. Admission stores two arrays, then a per-solve prefix descriptor
// and one range result per block. The reserved flag arena has 2*rows words.
struct SpsvBinary64ScratchView {
    __gm__ float* base;
    SpsvComplexWorkspaceGeometry geometry;
    uint32_t blocks;
    __gm__ uint64_t* factors{(__gm__ uint64_t*)(base + geometry.binary64FactorWord)};
    __gm__ uint32_t* admission{(__gm__ uint32_t*)(base + geometry.binary64FlagWord)};
    __gm__ uint32_t* suffix{admission + blocks};
    __gm__ uint32_t* prefixLevels{suffix + blocks};
    __gm__ uint32_t* prefixRange{prefixLevels + 1};
    __gm__ uint64_t* saved{(__gm__ uint64_t*)(base + geometry.binary64SavedWord)};
};

#endif // SPSV_COMPLEX_WORKSPACE_H_
