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

#ifndef SPSV_KERNEL_H_
#define SPSV_KERNEL_H_

#include <cstdint>
#include "../shared/spsv_tiling_data.h"

// GM_ADDR: kernel global-memory pointer type. The #ifndef guard is a repo
// convention (also used by spmm/gpsv) so that the CANN SDK definition takes
// precedence when available. If the SDK defines GM_ADDR with a different
// underlying type, the SDK version wins and this fallback is skipped.
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void spsv_analysis_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace, uint32_t numBlocks,
                             const SpsvTilingData& tiling, void* stream);

void spsv_analysis_complex_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
                                     const SpsvTilingData& tiling, void* stream);

// Multi-core analysis split into three independent launches (serial / parallel
// / final). The serial and final phases launch a single block; the parallel
// phase launches numBlocks blocks. Same-stream ordering provides the cross-
// kernel GM visibility that the old in-kernel SyncAll barriers provided.
void spsv_analysis_serial_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR workspace,
                                    const SpsvTilingData& tiling, void* stream);

void spsv_analysis_parallel_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR workspace, uint32_t numBlocks,
                                      const SpsvTilingData& tiling, void* stream);

void spsv_analysis_final_kernel_do(GM_ADDR workspace, const SpsvTilingData& tiling, void* stream);

void spsv_complex_binary64_primary_do(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                      const SpsvTilingData&, void*, bool early = false);

void spsv_real_scaled_primary_do(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                  const SpsvTilingData&, void*);

void spsv_real_binary64_primary_do(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                  const SpsvTilingData&, void*);

void spsv_solve_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY, GM_ADDR workspace,
                          uint32_t numBlocks, const SpsvTilingData& tiling, void* stream);

void spsv_complex_dyadic_recovery_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX, GM_ADDR vecY,
                                            GM_ADDR workspace, const SpsvTilingData& tiling, void* stream);

void spsv_complex_rational_recovery_kernel_do(GM_ADDR rowPtr, GM_ADDR colInd, GM_ADDR values, GM_ADDR vecX,
                                              GM_ADDR vecY, GM_ADDR workspace, const SpsvTilingData& tiling,
                                              void* stream);

void spsv_update_values_kernel_do(GM_ADDR newValues, GM_ADDR workspace, uint32_t numBlocks,
                                  const SpsvTilingData& tiling, void* stream);

void spsv_update_diag_kernel_do(GM_ADDR newValues, GM_ADDR workspace, uint32_t numBlocks, const SpsvTilingData& tiling,
                                void* stream);

void spsv_update_diag_csr_kernel_do(GM_ADDR newValues, GM_ADDR values, GM_ADDR workspace, uint32_t numBlocks,
                                    const SpsvTilingData& tiling, void* stream);

void spsv_fill_zero_kernel_do(GM_ADDR vecY, uint32_t numBlocks, const SpsvTilingData& tiling, void* stream);

void spsv_scale_copy_kernel_do(GM_ADDR vecX, GM_ADDR vecY, uint32_t numBlocks, const SpsvTilingData& tiling,
                               void* stream);

void spsv_scale_inf_kernel_do(GM_ADDR vecX, GM_ADDR vecY, uint32_t numBlocks, const SpsvTilingData& tiling,
                              void* stream);

void spsv_copy_values_kernel_do(GM_ADDR src, GM_ADDR dst, uint32_t numBlocks, const SpsvTilingData& tiling,
                                void* stream);

#endif // SPSV_KERNEL_H_
