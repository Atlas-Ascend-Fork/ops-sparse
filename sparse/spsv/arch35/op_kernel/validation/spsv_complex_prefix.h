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
#ifndef SPSV_COMPLEX_PREFIX_H_
#define SPSV_COMPLEX_PREFIX_H_

#include "../common/spsv_simt_common.h"

// A block-affine solve for the exact-solution minus the solved-shadow defect.
// Original outputs are unchanged; only proven error enclosures may shrink.
// Each 16-row block maps the preceding boundary through a signed complex
// transform and a constant. Prefix composition reaches boundary zero in
// O(log(blocks)) stages. Outward radii cover every product and sum. Unsupported
// graphs and unsuccessful certificates retain the existing exact recovery.
#include "spsv_block_bound.h"

struct SpsvComplexCertificateData {
    __gm__ const int32_t* rowPtr;
    __gm__ const int32_t* colInd;
    __gm__ const SpsvComplex64* values;
    __gm__ const SpsvComplex64* rhs;
    __gm__ const int32_t* diagPtr;
    __gm__ const SpsvComplex64* vecY;
    __gm__ const float* wide;
    SpsvComplex64 alpha;
    bool forward;
    bool nonUnit;
};

__aicore__ inline SpsvComplexCertificateData SpsvMakeComplexCertificateData(GM_ADDR rowPtr, GM_ADDR colInd,
                                                                            GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                                                            GM_ADDR wide, GM_ADDR diagPtr,
                                                                            SpsvComplex64 alpha, int32_t forward,
                                                                            int32_t nonUnit)
{
    return {(__gm__ const int32_t*)rowPtr,
            (__gm__ const int32_t*)colInd,
            (__gm__ const SpsvComplex64*)values,
            (__gm__ const SpsvComplex64*)rhs,
            (__gm__ const int32_t*)diagPtr,
            (__gm__ const SpsvComplex64*)y,
            (__gm__ const float*)wide,
            alpha,
            forward != 0,
            nonUnit != 0};
}

#include "spsv_complex_defect.h"

constexpr int32_t kSpsvPrefixWidth = 16;
constexpr int32_t kSpsvPrefixColumns = kSpsvPrefixWidth + 1;
constexpr int32_t kSpsvPrefixEntries = kSpsvPrefixWidth * kSpsvPrefixColumns;
constexpr int32_t kSpsvPrefixWords = 5;

struct SpsvPrefixValue {
    SpsvDF re;
    SpsvDF im;
    float radius;
};

__simt_callee__ inline SpsvPrefixValue SpsvPrefixZero() { return {{0.f, 0.f}, {0.f, 0.f}, 0.f}; }

__simt_callee__ inline bool SpsvPrefixIsZero(SpsvPrefixValue a)
{
    return a.re.hi == 0.f && a.re.lo == 0.f && a.im.hi == 0.f && a.im.lo == 0.f && a.radius == 0.f;
}

__simt_callee__ inline SpsvPrefixValue SpsvPrefixLoad(__gm__ const float* p)
{
    return {{p[0], p[1]}, {p[2], p[3]}, p[4]};
}

__simt_callee__ inline void SpsvPrefixStore(__gm__ float* p, SpsvPrefixValue a)
{
    p[0] = a.re.hi;
    p[1] = a.re.lo;
    p[2] = a.im.hi;
    p[3] = a.im.lo;
    p[4] = a.radius;
}

__simt_callee__ inline float SpsvPrefixNorm(SpsvPrefixValue a)
{
    return SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvAbsF(a.re.hi), SpsvAbsF(a.re.lo)),
                             SpsvFloatBoundAdd(SpsvAbsF(a.im.hi), SpsvAbsF(a.im.lo)));
}

__simt_callee__ inline SpsvPrefixValue SpsvPrefixAdd(SpsvPrefixValue a, SpsvPrefixValue b)
{
    if (SpsvPrefixIsZero(a)) {
        return b;
    }
    if (SpsvPrefixIsZero(b)) {
        return a;
    }
    float er = 0.f, ei = 0.f;
    SpsvBlockAcc(a.re, er, 1.f, b.re);
    SpsvBlockAcc(a.im, ei, 1.f, b.im);
    a.radius = SpsvFloatBoundAdd(SpsvFloatBoundAdd(a.radius, b.radius),
                                 SpsvFloatBoundUp((er + ei) * 0x1p-20f + 0x1p-108f));
    return a;
}

__simt_callee__ inline SpsvPrefixValue SpsvPrefixMul(SpsvPrefixValue a, SpsvPrefixValue b)
{
    if (SpsvPrefixIsZero(a) || SpsvPrefixIsZero(b)) {
        return SpsvPrefixZero();
    }
    SpsvPrefixValue z = SpsvPrefixZero();
    float er = 0.f, ei = 0.f;
    SpsvBlockAcc(z.re, er, a.re.hi, b.re);
    SpsvBlockAcc(z.re, er, a.re.lo, b.re);
    SpsvBlockAcc(z.re, er, -a.im.hi, b.im);
    SpsvBlockAcc(z.re, er, -a.im.lo, b.im);
    SpsvBlockAcc(z.im, ei, a.re.hi, b.im);
    SpsvBlockAcc(z.im, ei, a.re.lo, b.im);
    SpsvBlockAcc(z.im, ei, a.im.hi, b.re);
    SpsvBlockAcc(z.im, ei, a.im.lo, b.re);
    z.radius = SpsvFloatBoundAdd(SpsvFloatBoundAdd(SpsvFloatBoundMul(SpsvPrefixNorm(a), b.radius),
                                                   SpsvFloatBoundMul(SpsvPrefixNorm(b), a.radius)),
                                 SpsvFloatBoundAdd(SpsvFloatBoundMul(2.f, SpsvFloatBoundMul(a.radius, b.radius)),
                                                   SpsvFloatBoundUp((er + ei) * 0x1p-20f + 0x1p-108f)));
    return z;
}

__simt_callee__ inline int64_t SpsvPrefixAddress(int64_t block, int32_t row, int32_t column)
{
    return (block * kSpsvPrefixEntries + row * kSpsvPrefixColumns + column) * kSpsvPrefixWords;
}

__simt_callee__ inline uint32_t SpsvPrefixValidateRows(const SpsvComplexCertificateData& data, int64_t m, int32_t block,
                                                       int32_t blocks, int32_t tid)
{
    uint32_t bad = 0U;
    // Validate the complete input graph independently of any numerical
    // cancellation in the proposed affine transforms.
    for (int64_t r = static_cast<int64_t>(block) * 32 + tid; r < m; r += static_cast<int64_t>(blocks) * 32) {
        bad |= !SpsvIsFiniteF(data.alpha.real) || !SpsvIsFiniteF(data.alpha.imag) ||
               !SpsvBlockFiniteNormal(data.rhs[r].real) || !SpsvBlockFiniteNormal(data.rhs[r].imag) ||
               !SpsvBlockFiniteNormal(data.vecY[r].real) || !SpsvBlockFiniteNormal(data.vecY[r].imag) ||
               !SpsvBlockFiniteNormal(data.wide[4 * r + 1]) || !SpsvBlockFiniteNormal(data.wide[4 * r + 3]) ||
               data.rowPtr[r + 1] - data.rowPtr[r] > 64;
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t distance = data.forward ? r - data.colInd[p] : static_cast<int64_t>(data.colInd[p]) - r;
            bad |= distance > kSpsvPrefixWidth || !SpsvBlockFiniteNormal(data.values[p].real) ||
                   !SpsvBlockFiniteNormal(data.values[p].imag);
        }
    }
    return bad;
}

__simt_callee__ inline void SpsvPrefixPrepareDefects(const SpsvComplexCertificateData& data, int64_t m, int64_t part,
                                                     int32_t tid, __gm__ float* buffer)
{
    // Each residual depends only on the immutable input and solved shadow,
    // so compute the sixteen row defects in parallel before the affine
    // recurrence. The block barrier publishes them to the constant lane.
    if (tid < kSpsvPrefixWidth) {
        int64_t logical = part * kSpsvPrefixWidth + tid;
        int64_t r = data.forward ? logical : m - 1 - logical;
        SpsvPrefixValue defectValue = SpsvPrefixZero();
        if (logical < m) {
            float defect[5];
            if (SpsvComplexTightDefect(data, r, defect)) {
                defectValue = {{defect[0], defect[1]}, {defect[2], defect[3]}, defect[4]};
            } else {
                defectValue.radius = __builtin_inff();
            }
        }
        SpsvPrefixStore(buffer + SpsvPrefixAddress(part, tid, kSpsvPrefixWidth), defectValue);
    }
}

__simt_callee__ inline SpsvPrefixValue SpsvPrefixPrepareRow(const SpsvComplexCertificateData& data, int64_t m,
                                                            int64_t part, int32_t row, int32_t tid,
                                                            __gm__ float* buffer)
{
    int64_t logical = part * kSpsvPrefixWidth + row;
    int64_t r = data.forward ? logical : m - 1 - logical;
    SpsvPrefixValue z = SpsvPrefixZero();
    if (logical < m) {
        if (tid == kSpsvPrefixWidth) {
            z = SpsvPrefixLoad(buffer + SpsvPrefixAddress(part, row, kSpsvPrefixWidth));
        }
        for (int64_t p = data.rowPtr[r]; p < data.rowPtr[r + 1]; ++p) {
            int64_t c = data.colInd[p], at = data.forward ? c : m - 1 - c;
            if (at >= logical) {
                continue;
            }
            SpsvPrefixValue source = SpsvPrefixZero();
            if (at >= part * kSpsvPrefixWidth) {
                source = SpsvPrefixLoad(
                    buffer + SpsvPrefixAddress(part, static_cast<int32_t>(at - part * kSpsvPrefixWidth), tid));
            } else if (at - (part - 1) * kSpsvPrefixWidth == tid) {
                source.re.hi = 1.f;
            } else if (at < (part - 1) * kSpsvPrefixWidth) {
                z.radius = __builtin_inff();
            }
            z = SpsvPrefixAdd(z,
                              SpsvPrefixMul({{-data.values[p].real, 0.f}, {-data.values[p].imag, 0.f}, 0.f}, source));
        }
    }
    return z;
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void SpsvPrefixPrepare(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* a, __gm__ const SpsvComplex64* rhs,
    __gm__ const SpsvComplex64* y, __gm__ const int32_t* dp, __gm__ float* wide, SpsvComplex64 alpha, int64_t m,
    bool forward, int32_t block, int32_t blocks, __gm__ float* buffer, __gm__ uint32_t* invalid)
{
    SpsvSimtRefresh(wide);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    uint32_t bad = 0U;
    SpsvComplexCertificateData data{rp, ci, a, rhs, dp, y, wide, alpha, forward, false};
    bad = SpsvPrefixValidateRows(data, m, block, blocks, tid);
    bad = asc_reduce_max(bad);
    if (tid == 0) {
        invalid[block] = bad;
    }
    int64_t count = (m + kSpsvPrefixWidth - 1) / kSpsvPrefixWidth;
    for (int64_t part = block; part < count; part += blocks) {
        SpsvPrefixPrepareDefects(data, m, part, tid, buffer);
        asc_syncthreads();
        if (tid >= kSpsvPrefixColumns) {
            continue;
        }
        for (int32_t row = 0; row < kSpsvPrefixWidth; ++row) {
            SpsvPrefixValue z = SpsvPrefixPrepareRow(data, m, part, row, tid, buffer);
            SpsvPrefixStore(buffer + SpsvPrefixAddress(part, row, tid), z);
        }
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(512) inline void SpsvPrefixCompose(__gm__ const float* input,
                                                                            __gm__ float* output, int64_t count,
                                                                            int32_t gap, int32_t block, int32_t blocks)
{
    SpsvSimtRefresh(input);
    for (int64_t entry = static_cast<int64_t>(block) * 512 + threadIdx.x; entry < count * kSpsvPrefixEntries;
         entry += static_cast<int64_t>(blocks) * 512) {
        int64_t part = entry / kSpsvPrefixEntries;
        int32_t row = static_cast<int32_t>((entry % kSpsvPrefixEntries) / kSpsvPrefixColumns);
        int32_t column = static_cast<int32_t>(entry % kSpsvPrefixColumns);
        SpsvPrefixValue z = SpsvPrefixZero();
        if (part < gap) {
            z = SpsvPrefixLoad(input + entry * kSpsvPrefixWords);
        } else if (column != kSpsvPrefixWidth && part < 2 * gap) {
            // A prefix that reaches block zero has no incoming boundary state.
            z = SpsvPrefixZero();
        } else {
            if (column == kSpsvPrefixWidth) {
                z = SpsvPrefixLoad(input + SpsvPrefixAddress(part, row, kSpsvPrefixWidth));
            }
            for (int32_t k = 0; k < kSpsvPrefixWidth; ++k) {
                z = SpsvPrefixAdd(z, SpsvPrefixMul(SpsvPrefixLoad(input + SpsvPrefixAddress(part, row, k)),
                                                   SpsvPrefixLoad(input + SpsvPrefixAddress(part - gap, k, column))));
            }
        }
        SpsvPrefixStore(output + entry * kSpsvPrefixWords, z);
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void SpsvPrefixPublish(__gm__ const float* input,
                                                                            __gm__ const SpsvComplex64* y,
                                                                            __gm__ float* wide, int64_t m, bool forward,
                                                                            int32_t block, int32_t blocks,
                                                                            __gm__ uint32_t* risks)
{
    SpsvSimtRefresh(input);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    __gm__ float* prep = wide + 6 * m;
    uint32_t risk = 0U;
    for (int64_t r = static_cast<int64_t>(block) * 128 + tid; r < m; r += static_cast<int64_t>(blocks) * 128) {
        int64_t at = forward ? r : m - 1 - r;
        SpsvPrefixValue z = SpsvPrefixLoad(input + SpsvPrefixAddress(at / kSpsvPrefixWidth,
                                                                     static_cast<int32_t>(at % kSpsvPrefixWidth),
                                                                     kSpsvPrefixWidth));
        float er = SpsvFloatBoundAdd(z.radius, SpsvFloatBoundAdd(SpsvAbsF(z.re.hi), SpsvAbsF(z.re.lo)));
        float ei = SpsvFloatBoundAdd(z.radius, SpsvFloatBoundAdd(SpsvAbsF(z.im.hi), SpsvAbsF(z.im.lo)));
        if (prep[18 * r + 9] >= 0.f && er < prep[18 * r + 9]) {
            prep[18 * r + 9] = er;
        }
        if (prep[18 * r + 10] >= 0.f && ei < prep[18 * r + 10]) {
            prep[18 * r + 10] = ei;
        }
        risk |= SpsvFloatNeedsDependencyRetry(y[r].real, wide[4 * r + 1], prep[18 * r + 9]);
        risk |= SpsvFloatNeedsDependencyRetry(y[r].imag, wide[4 * r + 3], prep[18 * r + 10]);
    }
    SpsvPublishBlockRisk128(risk, risks, block);
}

__aicore__ inline void SpsvPrefixBarrier()
{
    AscendC::CrossCoreSetFlag<0, PIPE_V>(0);
    AscendC::CrossCoreWaitFlag<0, PIPE_S>(0);
}

#endif // SPSV_COMPLEX_PREFIX_H_
