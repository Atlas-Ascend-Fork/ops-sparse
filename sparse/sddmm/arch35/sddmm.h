/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef SDDMM_H
#define SDDMM_H

#include <stdint.h>
#include <acl/acl_base_rt.h>

#ifndef __gm__
#define __gm__
#endif

// Order pair tags (orderX, orderY).
#define SDDMM_ORDER_RR 0
#define SDDMM_ORDER_RC 1
#define SDDMM_ORDER_CR 2
#define SDDMM_ORDER_CC 3

// Data type encoding for host-side kernel dispatch.
#define SDDMM_DTYPE_INVALID (-1)
#define SDDMM_DTYPE_FP32 0
#define SDDMM_DTYPE_FP16 1

static inline int32_t SddmmDataTypeFromAcl(aclDataType valueType)
{
    if (valueType == ACL_FLOAT) {
        return SDDMM_DTYPE_FP32;
    }
    if (valueType == ACL_FLOAT16) {
        return SDDMM_DTYPE_FP16;
    }
    return SDDMM_DTYPE_INVALID;
}

namespace SddmmBudget {
constexpr int32_t kReduceTmpBytes = 32 * 1024;
constexpr int32_t kStridedChunk = 1024;
constexpr int32_t kScalarSlotBytes = 32;
constexpr int32_t kStridedElemBytes = kScalarSlotBytes;
constexpr int32_t kUbAlignmentSlack = 4 * 1024;
constexpr int32_t kUbFixedOverhead = kReduceTmpBytes + kStridedChunk * (kStridedElemBytes + sizeof(int32_t)) +
                                     kUbAlignmentSlack;
} // namespace SddmmBudget

// Per-launch value snapshot. Device scalar addresses are consumed on the stream.
// The workspace retains its reserved tiling region for layout compatibility.
typedef struct SddmmTilingData {
    int32_t m;             // C 的行数
    int32_t n;             // C 的列数
    int32_t k;             // X/Y 的缩减维度
    int32_t ldx;           // X 的 leading dimension
    int32_t ldy;           // Y 的 leading dimension
    int32_t kTile;         // K 维度分块大小
    int32_t nTile;         // 行内非零元素批量大小
    int32_t reorderOffset; // byte offset to reorder table in workspace
    int32_t binEdgeOffset; // byte offset to row-bin edge table in workspace
    int32_t opX;           // 0 = NON_TRANSPOSE, 1 = TRANSPOSE
    int32_t opY;           // 0 = NON_TRANSPOSE, 1 = TRANSPOSE
    int32_t orderPair;     // SDDMM_ORDER_RR / RC / CR / CC
    int32_t dataType;      // SDDMM_DTYPE_FP32 / FP16
    float alphaHost;
    float betaHost;
    uint64_t alphaDevice;
    uint64_t betaDevice;
    int32_t scalarType; // SDDMM_DTYPE_FP32 / FP16 for DEVICE scalars
} SddmmTilingData;

// Workspace layout (all aligned to 64B):
//   [ 0      , tilingOff       )  : 64B header (reserved / padding)
//   [ tilingOff   , reorderOff )  : SddmmTilingData
//   [ reorderOff  , binEdgeOff )  : int32 reorder[m]    (logical row -> original row)
//   [ binEdgeOff  , endOff     )  : int32 bin_edge[bin_num + 1]
#define SDDMM_WS_HEADER_BYTES 64
#define SDDMM_WS_ALIGN 64

static inline int64_t sddmm_align_up(int64_t v, int64_t a)
{
    if (a != 0) {
        return ((v + a - 1) / a) * a;
    }
    return v;
}

#endif // SDDMM_H
