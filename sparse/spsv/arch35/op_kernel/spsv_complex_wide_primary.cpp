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

#include "common/spsv_complex_workspace.h"
#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "spsv_kernel.h"
#include "common/spsv_primary_status.h"
#include "validation/spsv_complex_certificate_pipeline.h"
#include "aclsparse_host_utils.h"

template <int32_t WIDTH, bool NON_UNIT>
__schedmode__(1) __global__ __aicore__
    void spsv_complex_wide_primary(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                   GM_ADDR workspace, const SpsvTilingData t, uint32_t riskBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvUseComplexBinary64Primary(t) && SpsvComplexPrimaryCompleted(workspace, t.m)) {
        return;
    }
    if (((__gm__ const SpsvTilingData*)workspace)->numLevels <= 1) {
        return;
    }
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx()),
            blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    SpsvComplex64 alpha{t.alpha, t.alphaImag};
    if (t.alphaDevicePtr) {
        auto p = (__gm__ const float*)t.alphaDevicePtr;
        alpha = {p[0], p[1]};
    }
    SpsvScanCompletedComplexRows((__gm__ const SpsvComplex64*)y,
                                 (__gm__ const float*)(workspace + t.yWideOffset), t.m, riskBlocks, block,
                                 blocks);
    SpsvComplexCertificateData data{(__gm__ const int32_t*)rp,
                                    (__gm__ const int32_t*)ci,
                                    (__gm__ const SpsvComplex64*)values,
                                    (__gm__ const SpsvComplex64*)rhs,
                                    (__gm__ const int32_t*)(workspace + t.diagPtrOffset),
                                    (__gm__ const SpsvComplex64*)y,
                                    (__gm__ const float*)(workspace + t.yWideOffset),
                                    alpha,
                                    (t.fillMode == 0) == (t.opA == 0),
                                    NON_UNIT};
    uint32_t n = riskBlocks;
    int32_t levels = ((__gm__ const SpsvTilingData*)workspace)->numLevels;
    SpsvComplexCertificatePipeline pipeline{data, t.m, n, block, blocks};
    n = pipeline.Run<WIDTH, true>(levels, (__gm__ const int32_t*)(workspace + t.levelPtrOffset),
                                  (__gm__ const int32_t*)(workspace + t.levelRowOffset));
    auto flags = SpsvComplexScratchView{(__gm__ float*)(workspace + t.yWideOffset), {static_cast<uint64_t>(t.m)}}.risk;
    for (uint32_t at = n + static_cast<uint32_t>(block); at < riskBlocks;
         at += static_cast<uint32_t>(blocks)) {
        flags[at] = 0U;
    }
}

void spsv_complex_wide_primary_do(GM_ADDR rp, GM_ADDR ci, GM_ADDR v, GM_ADDR x, GM_ADDR y, GM_ADDR w,
                                  const SpsvTilingData& t, uint32_t blocks, uint32_t risks, void* stream)
{
    if (t.diagType == 0 && t.nnz / t.m <= 12) {
        spsv_complex_wide_primary<12, true><<<blocks, nullptr, stream>>>(rp, ci, v, x, y, w, t, risks);
    } else if (t.diagType == 0) {
        spsv_complex_wide_primary<16, true><<<blocks, nullptr, stream>>>(rp, ci, v, x, y, w, t, risks);
    } else if (t.nnz / t.m <= 14) {
        spsv_complex_wide_primary<12, false><<<blocks, nullptr, stream>>>(rp, ci, v, x, y, w, t, risks);
    } else {
        spsv_complex_wide_primary<16, false><<<blocks, nullptr, stream>>>(rp, ci, v, x, y, w, t, risks);
    }
}
