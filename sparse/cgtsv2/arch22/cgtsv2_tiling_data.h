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

#ifndef CGTSV2_ARCH22_TILING_DATA_H_
#define CGTSV2_ARCH22_TILING_DATA_H_

#include <cstdint>

constexpr uint64_t kCgtsv2WorkspaceAlignment = 128;
constexpr uint64_t kCgtsv2ComplexBytes = 2 * sizeof(float);
constexpr uint64_t kCgtsv2StatusBytes = 16;
// Host admission and device allocation must use the same UB layout constants.
constexpr uint32_t kCgtsv2UbRowAlignment = 32 / sizeof(float);
constexpr uint32_t kCgtsv2FactorTileSteps = 2048;
// Factor: L/D/U/V complex arrays and uint32 pivots; solve adds one complex RHS array.
constexpr uint32_t kCgtsv2FactorUbBytesPerRow = 4 * kCgtsv2ComplexBytes + sizeof(uint32_t);
constexpr uint32_t kCgtsv2SolveUbBytesPerRow = 5 * kCgtsv2ComplexBytes + sizeof(uint32_t);
// Conservative dispatch reserve for TPipe/compiler-managed UB use, not a hardware capacity.
// Allocation is admitted only when the reported UB capacity also covers this reserve.
constexpr uint64_t kCgtsv2UbHeadroomBytes = 16 * 1024;
static_assert(kCgtsv2FactorUbBytesPerRow == 36 && kCgtsv2SolveUbBytesPerRow == 44,
    "UB row sizes must track the factor and solve arrays");
constexpr int32_t kCgtsv2JacobiMinRows = 1024;
constexpr uint32_t kCgtsv2JacobiCenterRows = 512;
// The complex off-norm upper bound and maxComp(D) lower bound certify
// ||D^-1(L+U)||_inf < 0.74; 1.351352 is strictly greater than 1/0.74.
constexpr float kCgtsv2JacobiDominance = 1.351352f;
// With |b_component| <= 16 and maxComp(D) >= 1,
// ||x*||_inf <= 16*sqrt(2)/(1-0.74). Eighty exact-arithmetic steps bound
// truncation by 0.74^80 * 16*sqrt(2)/(1-0.74) < 3.01e-9.
// FP32 roundoff is addressed separately: 40 initial steps, compensated
// original-equation residual, then 40 correction steps. Their dependency
// radius is at most 39+1+39=79; retain the 80-row original-B halo.
constexpr uint32_t kCgtsv2JacobiIters = 80;
constexpr uint32_t kCgtsv2JacobiStride = kCgtsv2JacobiCenterRows + 2 * kCgtsv2JacobiIters + 16;
static_assert(kCgtsv2JacobiCenterRows % 8 == 0, "Jacobi center boundaries must be cacheline aligned");
static_assert(kCgtsv2JacobiStride % 16 == 0, "Jacobi arrays must be cacheline aligned");
constexpr uint32_t kCgtsv2JacobiArrayCount = 47;
static_assert(kCgtsv2JacobiIters % 2U == 0, "Jacobi refinement uses two equal halves");
constexpr uint32_t kCgtsv2JacobiStorageBytes = kCgtsv2JacobiArrayCount * kCgtsv2JacobiStride * sizeof(float) +
    2 * kCgtsv2JacobiIters * sizeof(float);

// All offsets are byte offsets from the caller-owned workspace base.
// Each region starts at a 128-byte boundary. The factor kernel initializes
// every active factor, pivot and status entry before the solve kernel reads it.
struct Cgtsv2TilingData {
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t blockNum;
    int32_t useUb;
    uint64_t lOffset;      // complex64[m - 1]
    uint64_t dOffset;      // complex64[m]
    uint64_t uOffset;      // complex64[m - 1]
    uint64_t vOffset;      // complex64[m - 2]
    uint64_t pivotOffset;  // uint32_t[m]
    uint64_t statusOffset; // 16 internal bytes, not a public info output
    uint64_t rhsScratchOffset; // complex64[m], shared by serial residual correction
    uint64_t lLowOffset;       // complex64[m - 1], low significands
    uint64_t dLowOffset;       // complex64[m], low significands
    uint64_t uLowOffset;       // complex64[m - 1], low significands
    uint64_t lExponentOffset;  // int64_t[2 * (m - 1)], independent real/imaginary exponents
    uint64_t dExponentOffset;  // int64_t[2 * m]
    uint64_t uExponentOffset;  // int64_t[2 * (m - 1)]
    uint64_t rhsLowOffset;     // complex64[m], low significands for rhsScratchOffset
    uint64_t rhsExponentOffset; // int64_t[2 * m]
    uint64_t correctionHighOffset; // complex64[m]
    uint64_t correctionLowOffset;  // complex64[m]
    uint64_t correctionExponentOffset; // int64_t[2 * m]
    uint64_t workspaceBytes;
};

#endif // CGTSV2_ARCH22_TILING_DATA_H_
