/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included once per specialization namespace: intentionally no include guard.
#include "spmm_1d_variant_config.inc"
#if __DAV_C220_CUBE__
#include "spmm_1d_aic.inc"
#endif
#if __DAV_C220_VEC__
#if !SPMM_DIRECT_B_GATHER_AIC
#include "spmm_1d_aiv_pack.inc"
#endif
#include "spmm_1d_aiv.inc"
#endif
#ifdef SPMM_B_PACK_SUB_TILE_WAS_DEFAULT
#undef SPMM_B_PACK_SUB_TILE_WAS_DEFAULT
#undef SPMM_B_PACK_SUB_TILE
#endif
