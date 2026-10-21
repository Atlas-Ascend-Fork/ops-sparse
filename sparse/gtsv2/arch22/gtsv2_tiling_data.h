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

#ifndef GTSV2_ARCH22_TILING_DATA_H_
#define GTSV2_ARCH22_TILING_DATA_H_

#include <cstdint>

// DAV_2201 has 192 KiB UB per vector core. The direct solve uses
//   4*m floats (factor data) + m bytes (pivot flags) + m floats (one RHS),
// i.e. 21*m bytes.  m=8192 consumes 168 KiB and leaves 24 KiB headroom.
// The multi-RHS factor kernel uses the same maximum: its final float array
// holds the vector dominance-check temporary instead of an RHS. The fused
// single-RHS kernel reuses its RHS array before loading B (20*m bytes total).
// At 128..6144 rows the vector Jacobi path adds two guarded FP32 ping-pong
// arrays. With 32-byte allocation padding, the maximum is 178304 bytes at
// 6144 rows including pivot storage. Above 6144,
// the multi-RHS solve instead uses one six-array allocation (24*m bytes),
// reloading normalized B in place. Its m=8192 maximum is exactly 192 KiB.
constexpr int32_t kGtsv2UbMaxM = 8192;

constexpr int32_t kGtsv2FactorArrays = 4;
constexpr uint64_t kGtsv2PackedFactorBytesPerRow =
    static_cast<uint64_t>(kGtsv2FactorArrays) * sizeof(float) + sizeof(uint8_t);
// Five (two FP32 significands + int64 exponent) arrays, m original FP32 RHS values, and m pivot bytes.
// The same allocation also covers the legacy two-array residual scratch.
constexpr uint64_t kGtsv2RangeScratchBytesPerRow = 85;

struct Gtsv2TilingData {
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t blockNum;
    int32_t useUb;
};

#endif  // GTSV2_ARCH22_TILING_DATA_H_
