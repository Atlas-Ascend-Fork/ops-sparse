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

#ifndef SPSV_PRECONDITIONED_BORDER_H_
#define SPSV_PRECONDITIONED_BORDER_H_

#include "../common/spsv_simt_common.h"

__simt_callee__ inline void SpsvPreconditionInverseDot(const SpsvComplexCertificateData& data, int64_t m, int64_t limit,
                                                       int32_t row, int32_t column, int32_t lane,
                                                       __ubuf__ const float* real, __ubuf__ const float* imag,
                                                       float& re, float& im)
{
    int64_t logical = limit + row, r = data.forward ? logical : m - 1 - logical;
    if (row > column) {
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t at = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
            if (at < limit + column || at >= logical) {
                continue;
            }
            int32_t k = static_cast<int32_t>(at - limit);
            re = fmaf(-data.values[p].real, real[k], re);
            re = fmaf(data.values[p].imag, imag[k], re);
            im = fmaf(-data.values[p].real, imag[k], im);
            im = fmaf(-data.values[p].imag, real[k], im);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvPreconditionInverse(const SpsvComplexCertificateData data,
                                                                                 int64_t m, int64_t limit,
                                                                                 int32_t block, int32_t blocks,
                                                                                 __gm__ float* matrix)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    int32_t n = static_cast<int32_t>(m - limit);
    __ubuf__ float real[320], imag[320];
    for (int32_t column = block; column < n; column += blocks) {
        for (int32_t row = lane; row < column; row += 32) {
            matrix[2 * (static_cast<int64_t>(row) * n + column)] = 0.f;
            matrix[2 * (static_cast<int64_t>(row) * n + column) + 1] = 0.f;
        }
        for (int32_t row = column; row < n; ++row) {
            float re = 0.f, im = 0.f;
            int64_t logical = limit + row;
            int64_t r = data.forward ? logical : m - 1 - logical;
            SpsvPreconditionInverseDot(data, m, limit, row, column, lane, real, imag, re, im);
            re = asc_reduce_add(re);
            im = asc_reduce_add(im);
            if (lane == 0) {
                if (row == column) {
                    re = 1.f;
                }
                if (row >= column) {
                    auto p = data.wide + 6 * m + 18 * r;
                    float a = re * p[11] - im * p[13];
                    float b = re * p[13] + im * p[11];
                    re = a;
                    im = b;
                }
                real[row] = re;
                imag[row] = im;
                matrix[2 * (static_cast<int64_t>(row) * n + column)] = re;
                matrix[2 * (static_cast<int64_t>(row) * n + column) + 1] = im;
            }
            asc_syncthreads();
        }
    }
    asc_threadfence();
}

__simt_callee__ inline void SpsvPreconditionAcc(float& hi, float& lo, float& cost, float a, float b)
{
    float product = a * b;
    float residual = fmaf(a, b, -product);
    auto head = SpsvTwoSum(hi, product);
    float middle = head.lo + residual;
    float low = lo + middle;
    cost += SpsvAbsF(middle) + SpsvAbsF(low);
    hi = head.hi;
    lo = low;
}

__simt_callee__ inline void SpsvPreconditionMerge(float& hi, float& lo, float& cost, int32_t lane)
{
    for (int32_t mask = 16; mask > 0; mask /= 2) {
        float rh = asc_shfl_xor(hi, mask), rl = asc_shfl_xor(lo, mask);
        float other = asc_shfl_xor(cost, mask);
        if (lane < mask) {
            auto head = SpsvTwoSum(hi, rh);
            float low = lo + rl;
            float middle = low + head.lo;
            cost += other + SpsvAbsF(low) + SpsvAbsF(middle);
            hi = head.hi;
            lo = middle;
        }
    }
}

__simt_callee__ inline float SpsvPreconditionCheckEntry(const SpsvComplexCertificateData& data, int64_t m,
                                                        int64_t limit, int64_t entry, int32_t lane,
                                                        __gm__ const float* matrix)
{
    int32_t n = static_cast<int32_t>(m - limit);
    int32_t row = static_cast<int32_t>(entry / n), column = static_cast<int32_t>(entry % n);
    // B and the explicitly initialized R are lower triangular.
    // Entries above the diagonal and products with at < column are
    // exact structural zeros, so they require no floating arithmetic.
    if (column > row) {
        return 0.f;
    }
    int64_t logical = limit + row, r = data.forward ? logical : m - 1 - logical;
    float rh = lane == 0 && row == column ? 1.f : 0.f;
    float rl = 0.f, ih = 0.f, il = 0.f, cost = 0.f;
    uint32_t bad = 0U;
    for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
        int64_t at = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
        if (at < limit + column || at > logical) {
            continue;
        }
        int64_t offset = 2 * ((at - limit) * n + column);
        float ar = data.values[p].real, ai = data.values[p].imag;
        float br = matrix[offset], bi = matrix[offset + 1];
        bad |= !SpsvBlockFiniteNormal(br) || !SpsvBlockFiniteNormal(bi);
        bad |= SpsvAbsF(ar) > 0x1p24f || SpsvAbsF(ai) > 0x1p24f || SpsvAbsF(br) > 0x1p24f || SpsvAbsF(bi) > 0x1p24f;
        SpsvPreconditionAcc(rh, rl, cost, -ar, br);
        SpsvPreconditionAcc(rh, rl, cost, ai, bi);
        SpsvPreconditionAcc(ih, il, cost, -ar, bi);
        SpsvPreconditionAcc(ih, il, cost, -ai, br);
    }
    // Two separate cost reductions would count some costs twice, which is
    // conservative. Keep their shared overestimate in the certificate.
    float rc = cost, ic = cost;
    SpsvPreconditionMerge(rh, rl, rc, lane);
    SpsvPreconditionMerge(ih, il, ic, lane);
    bad = asc_reduce_max(bad);
    if (lane == 0) {
        float count = static_cast<float>(data.rowPtr[r + 1] - data.rowPtr[r] + 64);
        float radius = SpsvFloatBoundUp((rc + ic) * (1.f + count * 0x1p-19f) * 0x1p-20f + count * 0x1p-120f);
        float norm = SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(rh), SpsvAbsF(rl)),
                                       SpsvFloatBoundAdd(SpsvAbsF(ih), SpsvAbsF(il)));
        return bad ? __builtin_inff() : SpsvFloatBoundAdd(norm, radius);
    }
    return 0.f;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvPreconditionCheck(const SpsvComplexCertificateData data,
                                                                                int64_t m, int64_t limit, int32_t block,
                                                                                int32_t blocks,
                                                                                __gm__ const float* matrix,
                                                                                __gm__ float* qmatrix)
{
    SpsvSimtRefresh(matrix);
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int32_t n = static_cast<int32_t>(m - limit);
    for (int64_t entry = static_cast<int64_t>(block) * 8 + warp; entry < static_cast<int64_t>(n) * n;
         entry += static_cast<int64_t>(blocks) * 8) {
        float bound = SpsvPreconditionCheckEntry(data, m, limit, entry, lane, matrix);
        if (lane == 0) {
            qmatrix[entry] = bound;
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvPreconditionNorms(int32_t n, int32_t block, int32_t blocks,
                                                                               __gm__ const float* matrix,
                                                                               __gm__ const float* qmatrix,
                                                                               __gm__ float* metrics)
{
    SpsvSimtRefresh(matrix);
    float q = 0.f, norm = 0.f;
    for (int32_t row = block * 32 + threadIdx.x; row < n; row += blocks * 32) {
        float a = 0.f, b = 0.f;
        for (int32_t column = 0; column <= row; ++column) {
            int64_t at = static_cast<int64_t>(row) * n + column;
            a = SpsvFloatBoundAdd(a, qmatrix[at]);
            b = SpsvFloatBoundAdd(b, SpsvFloatBoundAdd(SpsvAbsF(matrix[2 * at]), SpsvAbsF(matrix[2 * at + 1])));
        }
        // A failed row must remain unbounded through every later maximum.
        // In particular, a subsequent finite row must never erase a NaN.
        a = SpsvIsFiniteF(a) && a >= 0.f ? a : __builtin_inff();
        b = SpsvIsFiniteF(b) && b >= 0.f ? b : __builtin_inff();
        q = q > a ? q : a;
        norm = norm > b ? norm : b;
    }
    q = asc_reduce_max(q);
    norm = asc_reduce_max(norm);
    if (threadIdx.x == 0) {
        metrics[3 * block] = q;
        metrics[3 * block + 1] = norm;
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvPreconditionConstants(
    const SpsvComplexCertificateData data, int64_t m, int64_t limit, int32_t block, int32_t blocks,
    __gm__ const float* defects, __gm__ const float* inverses, __gm__ const float* corrections, __gm__ float* constants)
{
    SpsvSimtRefresh(corrections);
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t at = limit + static_cast<int64_t>(block) * 8 + warp; at < m; at += static_cast<int64_t>(blocks) * 8) {
        int64_t r = data.forward ? at : m - 1 - at;
        auto z = SpsvTriplePrefixZero();
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
            if (source < limit) {
                z = SpsvTriplePrefixAdd(
                    z, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag,
                                                        SpsvTriplePrefixLoad(corrections + source * 7)));
            }
        }
        z = SpsvTailWarpSum(z, lane);
        if (lane == 0) {
            auto d = SpsvCertificateDiagonal(data, r);
            auto defect = SpsvTriplePrefixLoad(defects + (at - limit) * 7);
            // defects are divided by d. Multiplying back encloses the exact
            // original residual; its independent inverse uncertainty remains.
            defect = SpsvTriplePrefixMulComplexScalar(d.real, d.imag, defect);
            z = SpsvTriplePrefixAdd(z, defect);
            SpsvTriplePrefixStore(constants + (at - limit) * 7, z);
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvPreconditionPropose(int64_t m, int64_t limit,
                                                                                  int32_t block, int32_t blocks,
                                                                                  __gm__ const float* matrix,
                                                                                  __gm__ const float* constants,
                                                                                  __gm__ float* corrections)
{
    SpsvSimtRefresh(constants);
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    int32_t n = static_cast<int32_t>(m - limit);
    for (int32_t row = block * 8 + warp; row < n; row += blocks * 8) {
        auto z = SpsvTriplePrefixZero();
        for (int32_t column = lane; column < n; column += 32) {
            auto c = SpsvTriplePrefixLoad(constants + 7 * column);
            c.radius = 0.f;
            int64_t at = 2 * (static_cast<int64_t>(row) * n + column);
            z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixMulComplexScalar(matrix[at], matrix[at + 1], c));
        }
        z = SpsvTailWarpSum(z, lane);
        if (lane == 0) {
            z.radius = 0.f;
            SpsvTriplePrefixStore(corrections + 7 * (limit + row), z);
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void SpsvPreconditionResidual(
    const SpsvComplexCertificateData data, int64_t m, int64_t limit, int32_t block, int32_t blocks,
    __gm__ const float* constants, __gm__ const float* corrections, __gm__ float* residuals)
{
    SpsvSimtRefresh(corrections);
    int32_t lane = static_cast<int32_t>(threadIdx.x) % 32, warp = static_cast<int32_t>(threadIdx.x) / 32;
    for (int64_t at = limit + static_cast<int64_t>(block) * 8 + warp; at < m; at += static_cast<int64_t>(blocks) * 8) {
        int64_t r = data.forward ? at : m - 1 - at;
        auto z = SpsvTriplePrefixZero();
        for (int64_t p = data.rowPtr[r] + lane; p < data.rowPtr[r + 1]; p += 32) {
            int64_t source = data.forward ? data.colInd[p] : m - 1 - data.colInd[p];
            if (source < limit || source > at) {
                continue;
            }
            auto c = SpsvTriplePrefixLoad(corrections + 7 * source);
            c.radius = 0.f;
            z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixMulComplexScalar(-data.values[p].real, -data.values[p].imag, c));
        }
        z = SpsvTailWarpSum(z, lane);
        if (lane == 0) {
            z = SpsvTriplePrefixAdd(z, SpsvTriplePrefixLoad(constants + 7 * (at - limit)));
            residuals[at - limit] = SpsvFloatBoundAdd(SpsvTriplePrefixNorm(z), z.radius);
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvPreconditionRadius(int64_t m, int64_t limit,
                                                                                 int32_t block, int32_t blocks,
                                                                                 __gm__ const float* residuals,
                                                                                 float inverseNorm,
                                                                                 __gm__ float* corrections)
{
    SpsvSimtRefresh(residuals);
    float residual = 0.f;
    for (int64_t row = 0; row < m - limit; ++row) {
        float value = residuals[row];
        value = SpsvIsFiniteF(value) && value >= 0.f ? value : __builtin_inff();
        residual = residual > value ? residual : value;
    }
    float bound = SpsvFloatBoundMul(SpsvFloatBoundMul(4.f, inverseNorm), residual);
    for (int64_t row = limit + static_cast<int64_t>(block) * 128 + threadIdx.x; row < m;
         row += static_cast<int64_t>(blocks) * 128) {
        corrections[7 * row + 6] = bound;
    }
    asc_threadfence();
}

#endif
