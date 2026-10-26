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
#ifndef SDDMM_HOST_TILING_H_
#define SDDMM_HOST_TILING_H_

#include "../sddmm_tiling_fields.h"

struct SddmmTilingData : SddmmTilingFields {};

struct SddmmBsrHostTiling {
    uint32_t m, n, k, bd, blocks, first, count, yPitch, group;
    int32_t base;
    bool colMajor;
    float ar, ai, br, bi;
};

// Round a byte count up to a whole 32B block. Used for every workspace region so
// the kernels can reach it with plain DataCopy/DataCopyPad.
inline uint64_t SddmmAlign32(uint64_t value) { return (value + 31u) & ~static_cast<uint64_t>(31u); }

// ---------------------------------------------------------------------------
// B-operand transpose (bulk path)
// ---------------------------------------------------------------------------
//
// Y is handed to SDDMM row-major, so one *column* of B -- exactly what a single
// dot product needs -- is spread over K rows of stride ld. Staging such a column
// costs one 32B DMA block per 2..4 useful bytes, which is where the scalar path
// spends essentially all of its time. Laying Y out as Yt[j * pitch + k] makes a
// column contiguous, and then one multi-block DMA per run of consecutive columns
// brings in a whole, fully packed tile.
//
// Yt has `cols` rows of SddmmYtPitch(k) elements each; the pad is never read
// (the kernel only ever touches k < K of a staged row) but it keeps every row
// 32B-aligned and makes the 64x64 transpose tiles line up with the row pitch.
constexpr uint32_t kYTransTile = 64;

inline uint32_t SddmmYtPitch(uint32_t k) { return (k + kYTransTile - 1u) / kYTransTile * kYTransTile; }

inline uint64_t SddmmYtBytes(uint32_t k, uint32_t cols, uint32_t elemSize)
{
    return SddmmAlign32(static_cast<uint64_t>(cols) * SddmmYtPitch(k) * elemSize);
}

// ---------------------------------------------------------------------------
// complex64 support
// ---------------------------------------------------------------------------
//
// arch22 has no complex vector type, and the arch35 answer to that
// (__simt_callee__ kernels) does not exist on dav-2201. So the complex rows of
// the support matrix are computed as two *ordinary fp32* SDDMM runs over a
// real-valued rewriting of the problem, followed by one recombination pass.
//
// Write X = Xr + i*Xi and Y = Yr + i*Yi. A complex64 array is just fp32 pairs,
// so Y viewed as fp32 is the [k, 2n] matrix Yf with Yf[t, 2j] = Yr[t,j] and
// Yf[t, 2j+1] = Yi[t,j] -- no repacking needed, the float view is the matrix.
// Splitting only X (which is m x k, i.e. tiny next to Y) gives
//
//     (Xr @ Yf)[i, 2j]   = sum_t Xr[i,t]*Yr[t,j] = A     (Xr @ Yf)[i, 2j+1] = C
//     (Xi @ Yf)[i, 2j]   = sum_t Xi[i,t]*Yr[t,j] = B     (Xi @ Yf)[i, 2j+1] = D
//
// and the answer is Re = A - D, Im = C + B. Both runs need the *doubled*
// sparsity pattern so that each logical entry (i,j) owns one result slot per
// real column, which is what sddmm_complex_prep_csr builds.
//
// A single workspace carries every intermediate; the three-stage API still
// exposes exactly one allocation.

// Byte offsets of the complex64 scratch regions. Every region starts on a 32B
// boundary so the kernels can use plain DataCopy/DataCopyPad on them.
struct SddmmComplexPlan {
    uint64_t xReal;  // fp32, rows * k elements, the real plane of X
    uint64_t xImag;  // fp32, rows * k elements, the imaginary plane of X
    uint64_t rowPtr; // (rows + 1) int32, doubled row offsets
    uint64_t colIdx; // 2 * nnz int32, one entry per real column
    uint64_t prodP;  // 2 * nnz fp32, the Xr run
    uint64_t prodQ;  // 2 * nnz fp32, the Xi run
    uint64_t yTrans; // fp32, [2 * cols, yPitch], the transposed float view of Y
    uint32_t yPitch;
    bool useYt;
    bool directOutput;
    uint64_t total;
};

inline SddmmComplexPlan SddmmComplexPlanOf(uint64_t rows, uint64_t k, uint64_t cols, uint64_t nnz)
{
    const uint64_t plane = SddmmAlign32(rows * k * sizeof(float));
    SddmmComplexPlan plan{};
    uint64_t offset = 0;
    plan.xReal = offset;
    offset += plane;
    plan.xImag = offset;
    offset += plane;
    // The transpose pays for itself when the sparsity pattern reuses Y at least
    // once on average. The SIMD kernel can encode at most 255 32B blocks in its
    // repeat stride, and the transpose kernel works on full 64-column tiles.
    const bool colsFit = cols <= (static_cast<uint64_t>(UINT32_MAX) / 2u);
    const uint64_t realCols = 2u * cols;
    if (colsFit && k >= 32u && realCols >= kYTransTile && nnz * k >= cols) {
        const uint32_t pitch = SddmmYtPitch(static_cast<uint32_t>(k));
        if (pitch <= 255u * 8u) {
            plan.useYt = true;
            plan.yPitch = pitch;
            plan.yTrans = offset;
            offset += SddmmYtBytes(static_cast<uint32_t>(k), static_cast<uint32_t>(realCols), sizeof(float));
        }
    }
    plan.directOutput = plan.useYt && k <= SddmmBudget::kComplexDirectKLimit;
    if (!plan.directOutput) {
        plan.rowPtr = offset;
        offset += SddmmAlign32((rows + 1u) * sizeof(int32_t));
        plan.colIdx = offset;
        offset += SddmmAlign32(2u * nnz * sizeof(int32_t));
        plan.prodP = offset;
        offset += SddmmAlign32(2u * nnz * sizeof(float));
        plan.prodQ = offset;
        offset += SddmmAlign32(2u * nnz * sizeof(float));
    }
    plan.total = offset;
    return plan;
}

#endif
