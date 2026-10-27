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

/*!
 * \file sddmm_kernel.h
 * \brief SDDMM kernel_do declarations shared by Host and Kernel code.
 */

#ifndef SDDMM_KERNEL_H_
#define SDDMM_KERNEL_H_

#include <cstdint>
#include "sddmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

extern "C" {

void sddmm_pattern_validate_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR workspace,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream);

void sddmm_pattern_query_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR workspace,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream);

void sddmm_pack_y_kernel_do(
    GM_ADDR matY, GM_ADDR packedY,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream);

void sddmm_kernel_do(
    GM_ADDR sparseOffsets, GM_ADDR sparseColumns, GM_ADDR sparseValues,
    GM_ADDR matX, GM_ADDR matY,
    const SddmmTilingData &tiling, uint32_t numBlocks, void *stream);

} // extern "C"

#endif // SDDMM_KERNEL_H_
