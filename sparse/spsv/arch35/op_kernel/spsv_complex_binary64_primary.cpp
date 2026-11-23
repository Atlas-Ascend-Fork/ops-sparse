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

#include "aclsparse_host_utils.h"

#include <cstdint>
#include "kernel_operator.h"
#include "c_api/vector_datamove/vector_datamove.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "spsv_kernel.h"
#include "math/spsv_arith.h"
#include "common/spsv_schedule_common.h"
#define SPSV_SOFT_NAMESPACE spsv_scalar
#define SPSV_SOFT __aicore__ __attribute__((always_inline)) inline
#define SPSV_SOFT_CLZ(value) __builtin_clzll(value)
#include "math/spsv_binary64.h"
#undef SPSV_BINARY64_H_
#undef SPSV_SOFT_NAMESPACE
#undef SPSV_SOFT
#undef SPSV_SOFT_CLZ
#include "math/spsv_binary64.h"
#include "common/spsv_primary_status.h"
#include "validation/spsv_complex_certificate_pipeline.h"
__simt_callee__ inline spsv_soft::Complex ReadSoft(SpsvComplex64 x)
{
    uint32_t re, im;
    __builtin_memcpy(&re, &x.real, 4);
    __builtin_memcpy(&im, &x.imag, 4);
    return {spsv_soft::From32(re), spsv_soft::From32(im)};
}
__simt_callee__ inline void WriteSoft(__gm__ SpsvComplex64* y, int64_t row, spsv_soft::Complex x)
{
    uint32_t re = spsv_soft::To32(x.re), im = spsv_soft::To32(x.im);
    float fr, fi;
    __builtin_memcpy(&fr, &re, 4);
    __builtin_memcpy(&fi, &im, 4);
    y[row].real = fr;
    y[row].imag = fi;
}

__simt_callee__ inline uint32_t FloatBits(float x)
{
    uint32_t result;
    __builtin_memcpy(&result, &x, 4);
    return result;
}

__simt_callee__ inline bool ReplayAdmitsCoefficient(SpsvComplex64 value, bool diagonal)
{
    uint32_t re = FloatBits(value.real) & 0x7fffffffU;
    uint32_t im = FloatBits(value.imag) & 0x7fffffffU;
    uint32_t magnitude = re > im ? re : im;
    // Keep extreme/singular systems on the established certified/exact path.
    // The soft arithmetic handles signs, zero components and exponents here.
    return magnitude <= 0x41000000U && (!diagonal || magnitude >= 0x33800000U);
}

__simt_callee__ inline bool ReplayCoefficientNormal(SpsvComplex64 value)
{
    uint32_t real = (FloatBits(value.real) >> 23) & 255U;
    uint32_t imag = (FloatBits(value.imag) >> 23) & 255U;
    return real - 1U < 130U && imag - 1U < 130U;
}

__simt_callee__ inline bool ReplayAdmitsRow(__gm__ const int32_t* rp, __gm__ const int32_t* ci,
                                            __gm__ const SpsvComplex64* values, __gm__ const int32_t* diag, int64_t m,
                                            int64_t row, bool forward, bool nonUnit, bool& predecessor, bool& normal)
{
    predecessor = row == (forward ? 0 : m - 1);
    int begin = rp[row], end = rp[row + 1];
    if (end < begin || (nonUnit && (diag[row] < begin || diag[row] >= end))) {
        return false;
    }
    for (int p = begin; p < end; ++p) {
        int64_t column = ci[p], distance = forward ? row - column : column - row;
        if (column < 0 || column >= m || distance < 0 || (p > begin && ci[p] <= ci[p - 1])) {
            return false;
        }
        if (!ReplayAdmitsCoefficient({values[p].real, values[p].imag}, nonUnit && column == row) ||
            (nonUnit && column == row && p != diag[row])) {
            return false;
        }
        if (column != row) {
            normal &= ReplayCoefficientNormal({values[p].real, values[p].imag});
        }
        predecessor |= distance == 1;
    }
    return !nonUnit || ci[diag[row]] == row;
}
__simt_callee__ inline void ReplayPublishAdmission(
    __ubuf__ const uint32_t* warp, __ubuf__ const uint32_t* suffix, uint32_t invalid, uint32_t suffixBegin,
    __gm__ uint32_t* flags, int32_t block, int32_t blocks)
{
    for (int32_t w = 1; w < 8; ++w) {
        invalid |= warp[w];
        suffixBegin = suffixBegin > suffix[w] ? suffixBegin : suffix[w];
    }
    flags[block] = invalid;
    flags[blocks + block] = suffixBegin;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ValidateReplay(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ const SpsvComplex64* y, __gm__ const float* wide,
    __gm__ const int32_t* diag, SpsvComplex64 alpha, int64_t m, bool forward, bool nonUnit, int32_t block,
    int32_t blocks, __gm__ uint32_t* flags, bool early)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)values);
    }
    asc_syncthreads();
    uint32_t invalid = 0, required = early ? 1U : 0U, suffixBegin = 0;
    bool normal = true;
    for (int64_t row = static_cast<int64_t>(block) * 256 + threadIdx.x; row < m;
         row += static_cast<int64_t>(blocks) * 256) {
        bool predecessor = false;
        invalid |= !ReplayAdmitsRow(rp, ci, values, diag, m, row, forward, nonUnit, predecessor, normal);
        if (!predecessor) {
            uint32_t after = static_cast<uint32_t>(forward ? row + 1 : m - row);
            suffixBegin = suffixBegin > after ? suffixBegin : after;
        }
        invalid |= (FloatBits(rhs[row].real) & 0x7fffffffU) > 0x41000000U;
        invalid |= (FloatBits(rhs[row].imag) & 0x7fffffffU) > 0x41000000U;
        __gm__ const float* prep = wide + 6 * m;
        // Certified overflow is a completed component, not a replay request.
        if (!early) {
            required |= SpsvFloatNeedsDependencyRetry(y[row].real, wide[4 * row + 1], prep[18 * row + 9]) ||
                        SpsvFloatNeedsDependencyRetry(y[row].imag, wide[4 * row + 3], prep[18 * row + 10]);
        }
    }
    invalid |= (FloatBits(alpha.real) & 0x7fffffffU) > 0x41000000U;
    invalid |= (FloatBits(alpha.imag) & 0x7fffffffU) > 0x41000000U;
    invalid = asc_reduce_max(invalid);
    required = asc_reduce_max(required);
    suffixBegin = asc_reduce_max(suffixBegin);
    uint32_t abnormal = asc_reduce_max(normal ? 0U : 1U);
    invalid |= (required << 1) | (abnormal << 2);
    __ubuf__ uint32_t warp[8], warpSuffix[8];
    int tid = static_cast<int>(threadIdx.x);
    if ((tid & 31) == 0) {
        warp[tid / 32] = invalid;
        warpSuffix[tid / 32] = suffixBegin;
    }
    asc_syncthreads();
    if (tid == 0) {
        ReplayPublishAdmission(warp, warpSuffix, invalid, suffixBegin, flags, block, blocks);
    }
    asc_threadfence();
}
extern "C" __schedmode__(1) __global__ __aicore__
    void spsv_replay_admission(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y, GM_ADDR ws,
                               const SpsvTilingData t, bool early)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    SpsvResolveComplexSolveCsr(rp, ci, values, ws, t);
    SpsvComplex64 alpha{t.alpha, t.alphaImag};
    if (t.alphaDevicePtr) {
        auto p = (__gm__ const float*)t.alphaDevicePtr;
        alpha = {p[0], p[1]};
    }
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (early && block == 0) {
        ((__gm__ SpsvTilingData*)ws)->primaryStatus = 0;
    }
    int32_t blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    uint32_t risks = t.numBlocks > static_cast<uint32_t>(blocks) ? t.numBlocks : static_cast<uint32_t>(blocks);
    auto data = SpsvMakeComplexCertificateData(rp, ci, values, rhs, y, ws + t.yWideOffset, ws + t.diagPtrOffset, alpha,
                                               (t.fillMode == 0) == (t.opA == 0), t.diagType == 0);
    if (!early) {
        bool pending = SpsvPrepareComplexRecovery(data, t.m, risks, block, blocks, true);
        if (pending) {
            SpsvComplexCertificatePipeline pipeline{data, t.m, risks, block, blocks};
            risks = pipeline.Run<kSpsvTriplePrefixWidth, false>(
                ((__gm__ const SpsvTilingData*)ws)->numLevels,
                (__gm__ const int32_t*)(ws + t.levelPtrOffset), (__gm__ const int32_t*)(ws + t.levelRowOffset));
            pending = SpsvComplexRecoveryHasRisk(data, t.m, risks, blocks);
        }
        if (!pending) {
            if (block == 0) {
                SpsvPublishComplexCertificateCompleted(ws);
            }
            return;
        }
    }
    // End the certificate arena lifetime before admission flags reuse it.
    if (blocks > 1) {
        SpsvTriplePrefixBarrier();
    }
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(t.m)};
    asc_vf_call<ValidateReplay>(
        dim3{256, 1, 1}, (__gm__ const int32_t*)rp, (__gm__ const int32_t*)ci, (__gm__ const SpsvComplex64*)values,
        (__gm__ const SpsvComplex64*)rhs, (__gm__ const SpsvComplex64*)y, (__gm__ const float*)(ws + t.yWideOffset),
        (__gm__ const int32_t*)(ws + t.diagPtrOffset), alpha, t.m, (t.fillMode == 0) == (t.opA == 0), t.diagType == 0,
        block, blocks, (__gm__ uint32_t*)(ws + t.yWideOffset + geometry.binary64FlagWord * sizeof(float)), early);
}

__aicore__ __attribute__((always_inline)) inline bool SpsvReplayFlagsRequireSolve(__gm__ const uint32_t* flags, uint32_t count)
{
    uint32_t summary = 0;
    for (uint32_t b = 0; b < count; ++b) {
        summary |= flags[b];
    }
    return (summary & 1U) == 0U && (summary & 2U) != 0U;
}
__simt_callee__ inline uint64_t Product32(uint32_t a, uint64_t b)
{
    return spsv_soft::Multiply32<false>(a, b);
}
__simt_callee__ __attribute__((always_inline)) inline uint64_t ReplayNormalProduct(
    uint32_t a, uint64_t b)
{
    return spsv_soft::MultiplyNormal32(a, b);
}

__simt_callee__ __attribute__((always_inline)) inline bool ReplayNormalProductSafe(uint32_t a, uint64_t b)
{
    uint32_t ea = (a >> 23) & 255U;
    uint32_t eb = static_cast<uint32_t>((b >> 52) & 2047U);
    return ea - 1U < 130U && eb - 128U < 1914U;
}

__simt_callee__ inline uint64_t ProductPart(SpsvComplex64 a, spsv_soft::Complex b, int component)
{
    if (spsv_soft::IsInf(b.re) || spsv_soft::IsInf(b.im)) {
        auto product = spsv_soft::Product(ReadSoft(a), b);
        return component ? product.im : product.re;
    }
    uint32_t ar = FloatBits(a.real), ai = FloatBits(a.imag);
    uint64_t br = component ? b.im : b.re, bi = component ? b.re : b.im;
    uint64_t first, second;
    if (ReplayNormalProductSafe(ar, br) && ReplayNormalProductSafe(ai, bi)) {
        first = ReplayNormalProduct(ar, br);
        second = ReplayNormalProduct(ai, bi);
    } else {
        first = Product32(ar, br);
        second = Product32(ai, bi);
    }
    return component ? spsv_soft::Add(first, second) : spsv_soft::Sub(first, second);
}

__simt_callee__ inline uint64_t PreparedReciprocal(uint64_t b)
{
    using namespace spsv_soft;
    if (IsZero(b) || IsNaN(b) || IsInf(b)) {
        return 0;
    }
    int eb;
    uint64_t mb;
    Unpack(b, eb, mb);
    return ReciprocalMantissa(mb);
}
__simt_callee__ inline uint64_t DivPrepared(uint64_t a, uint64_t b, uint64_t reciprocal)
{
    return spsv_soft::DivisionCore<true>(a, b, reciprocal);
}
__simt_callee__ inline uint64_t QuotientPart(spsv_soft::Complex a, SpsvComplex64 diagonal, int component,
                                             __gm__ const uint64_t* factors, int64_t row)
{
    using namespace spsv_soft;
    uint64_t ratio = factors[5 * row], denominator = factors[5 * row + 1];
    uint64_t ar = a.re & ~spsv_soft::Sign, ai = a.im & ~spsv_soft::Sign;
    if ((ar < Hidden && ai < 0x7c9fffffffffffffULL) || (ai < Hidden && ar < 0x7c9fffffffffffffULL)) {
        constexpr uint64_t scale = 0x4330000000000000ULL;
        a = {Mul(a.re, scale), Mul(a.im, scale)};
        denominator = Mul(denominator, scale);
    }

    uint64_t numerator;
    if ((FloatBits(diagonal.real) & 0x7fffffffU) < (FloatBits(diagonal.imag) & 0x7fffffffU)) {
        numerator = component ? Sub(Mul(a.im, ratio), a.re) : Add(Mul(a.re, ratio), a.im);
    } else {
        numerator = component ? Sub(a.im, Mul(a.re, ratio)) : Add(Mul(a.im, ratio), a.re);
    }
    return DivPrepared(numerator, denominator, factors[5 * row + 2]);
}
__simt_vf__ __aicore__ __launch_bounds__(256) inline void PrepareFactors(
    __gm__ const SpsvComplex64* values, __gm__ const int32_t* diag, __gm__ uint64_t* factors,
    __gm__ const SpsvComplex64* rhs, SpsvComplex64 alpha, bool nonUnit, int64_t m, int64_t start, int64_t stride)
{
    using namespace spsv_soft;
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)values);
    }
    asc_syncthreads();
    for (int64_t row = start + threadIdx.x; row < m; row += stride) {
        SpsvComplex64 d = nonUnit ? SpsvComplex64{values[diag[row]].real, values[diag[row]].imag} :
                                    SpsvComplex64{1.0f, 0.0f};
        auto b = ReadSoft(d);
        uint64_t ratio, denominator;
        if ((b.re & ~spsv_soft::Sign) < (b.im & ~spsv_soft::Sign)) {
            ratio = Div(b.re, b.im);
            denominator = Add(Product32(FloatBits(d.real), ratio), b.im);
        } else {
            ratio = Div(b.im, b.re);
            denominator = Add(Product32(FloatBits(d.imag), ratio), b.re);
        }
        factors[5 * row] = ratio;
        factors[5 * row + 1] = denominator;
        factors[5 * row + 2] = PreparedReciprocal(denominator);
        auto initial = spsv_soft::Product(ReadSoft(alpha), ReadSoft({rhs[row].real, rhs[row].imag}));
        factors[5 * row + 3] = initial.re;
        factors[5 * row + 4] = initial.im;
    }
    asc_threadfence();
}
extern "C" __global__ __aicore__ void spsv_soft64_prepare(GM_ADDR values, GM_ADDR rhs, GM_ADDR ws,
                                                          const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvComplexCertificateCompleted(ws, tiling.m)) {
        return;
    }
    if (tiling.opA != 0 && tiling.transValuesOffset >= 0) {
        values = ws + tiling.transValuesOffset;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(ws);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        state);
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(tiling.m)};
    auto flags = (__gm__ const uint32_t*)(ws + tiling.yWideOffset + geometry.binary64FlagWord * sizeof(float));
    if (!SpsvReplayFlagsRequireSolve(flags, AscendC::GetBlockNum())) {
        return;
    }
    SpsvComplex64 alpha{tiling.alpha, tiling.alphaImag};
    if (tiling.alphaDevicePtr) {
        auto a = (__gm__ const float*)tiling.alphaDevicePtr;
        alpha = {a[0], a[1]};
    }
    asc_vf_call<PrepareFactors>(
        dim3{256, 1, 1}, (__gm__ const SpsvComplex64*)values, (__gm__ const int32_t*)(ws + tiling.diagPtrOffset),
        (__gm__ uint64_t*)(ws + tiling.yWideOffset + geometry.binary64FactorWord * sizeof(float)),
        (__gm__ const SpsvComplex64*)rhs, alpha, tiling.diagType == 0, tiling.m,
        static_cast<int64_t>(AscendC::GetBlockIdx()) * 256, static_cast<int64_t>(AscendC::GetBlockNum()) * 256);
}
__simt_callee__ inline uint64_t ReplayProductTerm(__gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
                                                  __gm__ const uint64_t* saved, spsv_soft::Complex kept, int64_t m,
                                                  int64_t row, bool forward, int32_t p, int32_t end, int32_t component)
{
    using namespace spsv_soft;
    int32_t column = p < end ? ci[p] : static_cast<int32_t>(row);
    int64_t source = forward ? column : m - 1 - column;
    int64_t step = forward ? row : m - 1 - row;
    spsv_soft::Complex prior{asc_shfl(kept.re, source & 31, 32), asc_shfl(kept.im, source & 31, 32)};
    if (source < step && step - source >= 32) {
        prior = {saved[2 * column], saved[2 * column + 1]};
    }
    SpsvComplex64 coefficient{0.f, 0.f};
    uint64_t term = 0;
    if (p < end && source < step) {
        coefficient = {values[p].real, values[p].imag};
        term = ProductPart(coefficient, prior, component);
    }
    return term;
}

__simt_callee__ inline spsv_soft::Complex ReplayDivide(spsv_soft::Complex numerator, SpsvComplex64 diagonal,
                                                       int32_t lane, __gm__ const uint64_t* factors, int64_t row)
{
    using namespace spsv_soft;
    int32_t component = lane & 1;
    uint64_t answer = 0;
    if (lane < 2) {
        answer = QuotientPart(numerator, diagonal, component, factors, row);
    }
    spsv_soft::Complex solved{asc_shfl(answer, 0, 32), asc_shfl(answer, 1, 32)};
    if (IsNaN(solved.re) && IsNaN(solved.im) && (IsInf(numerator.re) || IsInf(numerator.im))) {
        numerator = {Box(numerator.re), Box(numerator.im)};
        SpsvComplex64 d{diagonal.real, -diagonal.imag};
        if (lane < 2) {
            answer = Mul(Infinity, ProductPart(d, numerator, component));
        }
        solved = {asc_shfl(answer, 0, 32), asc_shfl(answer, 1, 32)};
    }
    return solved;
}

__simt_callee__ inline spsv_soft::Complex ReplaySolveRow(spsv_soft::Complex numerator, bool nonUnit,
                                                         SpsvComplex64 diagonal, int32_t lane,
                                                         __gm__ const uint64_t* factors, int64_t row)
{
    using namespace spsv_soft;
    if (nonUnit) {
        return ReplayDivide(numerator, diagonal, lane, factors, row);
    }
    return numerator;
}

__simt_callee__ inline bool ReplayStopAtNaNSuffix(spsv_soft::Complex solved, uint64_t step, uint32_t suffixBegin,
                                                  int32_t lane, int64_t m, __gm__ uint64_t* saved)
{
    using namespace spsv_soft;
    if (!IsNaN(solved.re) || !IsNaN(solved.im) || step + 1 < suffixBegin) {
        return false;
    }
    if (lane == 0) {
        saved[2 * m] = step + 1;
    }
    return true;
}

__simt_callee__ inline spsv_soft::Complex ReplayNumerator(__gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
                                                          __gm__ const uint64_t* saved, __gm__ const uint64_t* factors,
                                                          spsv_soft::Complex kept, int64_t m, int64_t row, bool forward,
                                                          int32_t lane, int32_t begin, int32_t end,
                                                          int32_t diagonalIndex)
{
    using namespace spsv_soft;
    int32_t component = lane & 1;
    uint64_t sum = lane < 2 ? factors[5 * row + 3 + component] : 0;
    // Parallelize each complex product, but retain the original CSR subtraction
    // order across chunks. Unlike the old 32-row ring this also covers long rows.
    for (int32_t chunk = begin; chunk < end; chunk += 16) {
        uint64_t term = ReplayProductTerm(ci, values, saved, kept, m, row, forward, chunk + lane / 2, end, component);
        int32_t count = end - chunk < 16 ? end - chunk : 16;
        for (int32_t item = 0; item < count; ++item) {
            uint64_t partial = asc_shfl(term, 2 * item + component, 32);
            if (lane < 2 && chunk + item != diagonalIndex) {
                sum = Sub(sum, partial);
            }
        }
    }
    return {asc_shfl(sum, 0, 32), asc_shfl(sum, 1, 32)};
}

__simt_callee__ inline bool Soft64ReplayStep(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    spsv_soft::Complex& kept, int64_t m, bool forward, bool nonUnit,
    __gm__ const uint64_t* factors, __gm__ uint64_t* saved, __gm__ const int32_t* diag,
    int64_t step, uint32_t suffixBegin)
{
    using namespace spsv_soft;
    int32_t lane = static_cast<int32_t>(threadIdx.x), component = lane & 1;
    int64_t row = forward ? step : m - 1 - step;
    int32_t diagonalIndex = diag[row];
    int32_t begin = rp[row], end = rp[row + 1];
    int64_t distance = begin < end ? (forward ? row - ci[begin] : ci[end - 1] - row) : 0;
    if (distance >= 32) {
        asc_threadfence();
        asc_syncthreads();
    }
    auto numerator = ReplayNumerator(ci, values, saved, factors, kept, m, row, forward, lane, begin, end,
                                     diagonalIndex);
    SpsvComplex64 diagonal = nonUnit ? SpsvComplex64{values[diagonalIndex].real, values[diagonalIndex].imag} :
                                       SpsvComplex64{1.f, 0.f};
    auto solved = ReplaySolveRow(numerator, nonUnit, diagonal, lane, factors, row);
    if (lane < 2) {
        saved[2 * row + component] = component ? solved.im : solved.re;
    }
    if (lane == (step & 31)) {
        kept = solved;
    }
    return ReplayStopAtNaNSuffix(solved, static_cast<uint64_t>(step), suffixBegin, lane, m, saved);
}

__simt_callee__ inline void ReplaySoft64Rows(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    spsv_soft::Complex& kept, int64_t m, bool forward, bool nonUnit,
    __gm__ const uint64_t* factors, __gm__ uint64_t* saved, __gm__ const int32_t* diag, uint32_t suffixBegin)
{
    for (int64_t step = 0; step < m; ++step) {
        if (Soft64ReplayStep(rp, ci, values, kept, m, forward, nonUnit, factors, saved, diag, step, suffixBegin)) {
            break;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(32) inline void Soft64Replay(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    __gm__ const SpsvComplex64* rhs, __gm__ SpsvComplex64* y, int64_t m, bool forward, bool nonUnit,
    SpsvComplex64 alpha, __gm__ const int32_t* diag, __gm__ const uint64_t* factors, __gm__ uint64_t* saved,
    __gm__ SpsvTilingData* status, uint32_t suffixBegin)
{
    using namespace spsv_soft;
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)values);
    }
    asc_syncthreads();
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    spsv_soft::Complex kept{0, 0};
    if (lane == 0) {
        saved[2 * m] = static_cast<uint64_t>(m);
    }
    ReplaySoft64Rows(rp, ci, values, kept, m, forward, nonUnit, factors, saved, diag, suffixBegin);
    if (lane == 0) {
        status->primaryStatus = 2;
    }
    asc_threadfence();
}
extern "C" __global__ __aicore__ void spsv_soft64_replay(GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y,
                                                         GM_ADDR ws, const SpsvTilingData tiling, uint32_t flagCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvComplexCertificateCompleted(ws, tiling.m)) {
        return;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(ws);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        state);
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(tiling.m)};
    auto flags = (__gm__ const uint32_t*)(ws + tiling.yWideOffset + geometry.binary64FlagWord * sizeof(float));
    uint32_t count = flagCount;
    if (!SpsvReplayFlagsRequireSolve(flags, count)) {
        return;
    }
    uint32_t suffixBegin = 0;
    for (uint32_t block = 0; block < count; ++block) {
        uint32_t begin = flags[count + block];
        suffixBegin = suffixBegin > begin ? suffixBegin : begin;
    }
    SpsvComplex64 alpha{tiling.alpha, tiling.alphaImag};
    if (tiling.alphaDevicePtr) {
        auto a = (__gm__ const float*)tiling.alphaDevicePtr;
        alpha = {a[0], a[1]};
    }
    SpsvResolveComplexSolveCsr(rp, ci, values, ws, tiling);
    asc_vf_call<Soft64Replay>(
        dim3{32, 1, 1}, (__gm__ const int32_t*)rp, (__gm__ const int32_t*)ci, (__gm__ const SpsvComplex64*)values,
        (__gm__ const SpsvComplex64*)rhs, (__gm__ SpsvComplex64*)y, tiling.m,
        (tiling.fillMode == 0) == (tiling.opA == 0), tiling.diagType == 0, alpha,
        (__gm__ const int32_t*)(ws + tiling.diagPtrOffset),
        (__gm__ const uint64_t*)(ws + tiling.yWideOffset + geometry.binary64FactorWord * sizeof(float)),
        (__gm__ uint64_t*)(ws + tiling.yWideOffset + geometry.binary64SavedWord * sizeof(float)),
        (__gm__ SpsvTilingData*)ws, suffixBegin);
}

__aicore__ __attribute__((always_inline)) inline uint32_t ScalarFloatBits(float x)
{
    uint32_t result;
    __builtin_memcpy(&result, &x, 4);
    return result;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarProduct32(uint32_t a, uint64_t b)
{
    return spsv_scalar::Multiply32<true>(a, b);
}
__aicore__ __attribute__((always_inline)) inline uint64_t ScalarProductPart(SpsvComplex64 a, spsv_scalar::Complex b, int component)
{
    uint64_t first = ScalarProduct32(ScalarFloatBits(a.real), component ? b.im : b.re);
    uint64_t second = ScalarProduct32(ScalarFloatBits(a.imag), component ? b.re : b.im);
    return component ? spsv_scalar::Add(first, second) : spsv_scalar::Sub(first, second);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarDivPrepared(uint64_t a, uint64_t b, uint64_t reciprocal)
{
    return spsv_scalar::DivisionCore<true>(a, b, reciprocal);
}
__aicore__ __attribute__((always_inline)) inline uint64_t ScalarQuotientPart(spsv_scalar::Complex a, SpsvComplex64 diagonal, int component,
                                             __gm__ const uint64_t* factors, int64_t row)
{
    using namespace spsv_scalar;
    uint64_t ratio = factors[5 * row], denominator = factors[5 * row + 1];
    uint64_t ar = a.re & ~spsv_scalar::Sign, ai = a.im & ~spsv_scalar::Sign;
    if ((ar < Hidden && ai < 0x7c9fffffffffffffULL) || (ai < Hidden && ar < 0x7c9fffffffffffffULL)) {
        constexpr uint64_t scale = 0x4330000000000000ULL;
        a = {Mul(a.re, scale), Mul(a.im, scale)};
        denominator = Mul(denominator, scale);
    }

    uint64_t numerator;
    if ((ScalarFloatBits(diagonal.real) & 0x7fffffffU) < (ScalarFloatBits(diagonal.imag) & 0x7fffffffU)) {
        numerator = component ? Sub(Mul(a.im, ratio), a.re) : Add(Mul(a.re, ratio), a.im);
    } else {
        numerator = component ? Sub(a.im, Mul(a.re, ratio)) : Add(Mul(a.im, ratio), a.re);
    }
    return ScalarDivPrepared(numerator, denominator, factors[5 * row + 2]);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarNormalProduct(
    uint32_t a, uint64_t b)
{
    return spsv_scalar::MultiplyNormal32(a, b);
}

__aicore__ __attribute__((always_inline)) inline bool ScalarNormalProductSafe(uint32_t a, uint64_t b)
{
    uint32_t ea = (a >> 23) & 255U;
    uint32_t eb = static_cast<uint32_t>((b >> 52) & 2047U);
    return ea - 1U < 130U && eb - 128U < 1914U;
}

__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarReplayNormalProduct(
    SpsvComplex64 coefficient, spsv_scalar::Complex prior)
{
    using namespace spsv_scalar;
    uint32_t ar = ScalarFloatBits(coefficient.real), ai = ScalarFloatBits(coefficient.imag);
    return {AddNormal(ScalarNormalProduct(ar, prior.re), Neg(ScalarNormalProduct(ai, prior.im))),
            AddNormal(ScalarNormalProduct(ar, prior.im), ScalarNormalProduct(ai, prior.re))};
}

__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarReplayProduct(SpsvComplex64 coefficient, spsv_scalar::Complex prior)
{
    using namespace spsv_scalar;
    uint32_t ar = ScalarFloatBits(coefficient.real), ai = ScalarFloatBits(coefficient.imag);
    if (ScalarNormalProductSafe(ar, prior.re) && ScalarNormalProductSafe(ar, prior.im) &&
        ScalarNormalProductSafe(ai, prior.re) && ScalarNormalProductSafe(ai, prior.im)) {
        return {AddNormal(ScalarNormalProduct(ar, prior.re), Neg(ScalarNormalProduct(ai, prior.im))),
                AddNormal(ScalarNormalProduct(ar, prior.im), ScalarNormalProduct(ai, prior.re))};
    }
    spsv_scalar::Complex term{ScalarProductPart(coefficient, prior, 0), ScalarProductPart(coefficient, prior, 1)};
    if (IsNaN(term.re) && IsNaN(term.im) && (IsInf(prior.re) || IsInf(prior.im))) {
        prior = {Box(prior.re), Box(prior.im)};
        term = {Mul(Infinity, ScalarProductPart(coefficient, prior, 0)),
                Mul(Infinity, ScalarProductPart(coefficient, prior, 1))};
    }
    return term;
}

__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarReplayDivide(spsv_scalar::Complex numerator, SpsvComplex64 diagonal,
                                                          __gm__ const uint64_t* factors, int64_t row)
{
    using namespace spsv_scalar;
    spsv_scalar::Complex solved{ScalarQuotientPart(numerator, diagonal, 0, factors, row),
                   ScalarQuotientPart(numerator, diagonal, 1, factors, row)};
    if (IsNaN(solved.re) && IsNaN(solved.im) && (IsInf(numerator.re) || IsInf(numerator.im))) {
        numerator = {Box(numerator.re), Box(numerator.im)};
        SpsvComplex64 d{diagonal.real, -diagonal.imag};
        solved = {Mul(Infinity, ScalarProductPart(d, numerator, 0)),
                  Mul(Infinity, ScalarProductPart(d, numerator, 1))};
    }
    return solved;
}

__aicore__ inline int32_t ScalarStageEntries(__gm__ const int32_t* columns, __gm__ const SpsvComplex64* values,
                                            __ubuf__ int32_t* localColumns, __ubuf__ float* localValues,
                                            int32_t p, int64_t nnz)
{
    int32_t begin = p / 1024 * 1024;
    int32_t count = nnz - begin < 1024 ? static_cast<int32_t>(nnz - begin) : 1024;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    asc_copy_gm2ub_align(localColumns, (__gm__ int32_t*)(columns + begin), count * sizeof(int32_t));
    asc_copy_gm2ub_align(localValues, (__gm__ float*)(values + begin), count * sizeof(SpsvComplex64));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    return begin;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ReplayProductChunk(
    __ubuf__ const int32_t* columns, __ubuf__ const float* values, __ubuf__ const volatile uint64_t* saved,
    __ubuf__ volatile uint64_t* terms, int32_t offset, int32_t count)
{
    using namespace spsv_soft;
    int32_t lane = static_cast<int32_t>(threadIdx.x), component = lane & 1;
    for (int32_t item = lane / 2; item < count; item += 128) {
        int32_t index = offset + item, column = columns[index];
        spsv_soft::Complex prior{saved[2 * column], saved[2 * column + 1]};
        SpsvComplex64 coefficient{values[2 * index], values[2 * index + 1]};
        uint64_t term = ProductPart(coefficient, prior, component);
        terms[2 * item + component] = term;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ReplayNormalProductChunk(
    __ubuf__ const int32_t* columns, __ubuf__ const float* values, __ubuf__ const volatile uint64_t* saved,
    __ubuf__ volatile uint64_t* terms, int32_t offset, int32_t count)
{
    using namespace spsv_soft;
    int32_t lane = static_cast<int32_t>(threadIdx.x), component = lane & 1;
    for (int32_t item = lane / 2; item < count; item += 128) {
        int32_t index = offset + item, column = columns[index];
        uint32_t ar = FloatBits(values[2 * index]), ai = FloatBits(values[2 * index + 1]);
        uint64_t br = saved[2 * column + component], bi = saved[2 * column + 1 - component];
        uint64_t first = ReplayNormalProduct(ar, br), second = ReplayNormalProduct(ai, bi);
        terms[2 * item + component] = component ? AddNormal(first, second) : AddNormal(first, Neg(second));
    }
}

__aicore__ inline int32_t ReplayWidePrefix(__gm__ const int32_t* levels, int32_t count, int64_t m)
{
    int32_t prefix = SpsvComplexSingletonPrefix(levels, count, m);
    return prefix < count && static_cast<int64_t>(levels[prefix]) + count - prefix == m &&
           levels[prefix] >= 256 ? prefix : 0;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ReplayClearSaved(
    __gm__ uint64_t* saved, int64_t m, int32_t block, int32_t blocks)
{
    for (int64_t row = static_cast<int64_t>(block) * 256 + threadIdx.x; row < m;
         row += static_cast<int64_t>(blocks) * 256) {
        saved[2 * row] = 0;
        saved[2 * row + 1] = 0;
    }
    asc_threadfence();
}

__simt_callee__ inline spsv_soft::Complex ReplayPrefixRow(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    __gm__ const uint64_t* factors, __gm__ const uint64_t* saved, int32_t row, bool forward)
{
    using namespace spsv_soft;
    spsv_soft::Complex sum{factors[5 * row + 3], factors[5 * row + 4]};
    for (int32_t p = rp[row]; p < rp[row + 1]; ++p) {
        int32_t column = ci[p];
        if (forward ? column < row : column > row) {
            auto coefficient = ReadSoft({values[p].real, values[p].imag});
            auto term = spsv_soft::Product(coefficient, {saved[2 * column], saved[2 * column + 1]});
            sum = {Sub(sum.re, term.re), Sub(sum.im, term.im)};
        }
    }
    if (IsNaN(sum.re) || IsNaN(sum.im) || IsInf(sum.re) || IsInf(sum.im)) {
        sum = Quotient(sum, {One, 0});
    }
    return sum;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ReplayPrefixLevel(
    __gm__ const int32_t* rp, __gm__ const int32_t* ci, __gm__ const SpsvComplex64* values,
    __gm__ const uint64_t* factors, __gm__ uint64_t* saved, __gm__ const int32_t* levelPtr,
    __gm__ const int32_t* levelRow, int32_t level, bool forward, int32_t block, int32_t blocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)saved);
    }
    asc_syncthreads();
    int32_t begin = levelPtr[level], end = levelPtr[level + 1];
    for (int32_t at = begin + block * 256 + static_cast<int32_t>(threadIdx.x); at < end; at += blocks * 256) {
        int32_t row = levelRow[at];
        auto sum = ReplayPrefixRow(rp, ci, values, factors, saved, row, forward);
        saved[2 * row] = sum.re;
        saved[2 * row + 1] = sum.im;
    }
    asc_threadfence();
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ReplayPrefixRange(
    __gm__ const uint64_t* saved, __gm__ const int32_t* rows, int32_t count,
    __gm__ uint32_t* flags, int32_t block, int32_t blocks)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)saved);
    }
    asc_syncthreads();
    uint32_t abnormal = 0;
    for (int32_t at = block * 256 + static_cast<int32_t>(threadIdx.x); at < count; at += blocks * 256) {
        int32_t row = rows[at];
        abnormal |= !spsv_soft::ReplayOperandBounded(saved[2 * row]) ||
                    !spsv_soft::ReplayOperandBounded(saved[2 * row + 1]);
    }
    abnormal = asc_reduce_max(abnormal);
    __ubuf__ uint32_t warp[8];
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if ((tid & 31) == 0) {
        warp[tid / 32] = abnormal;
    }
    asc_syncthreads();
    if (tid == 0) {
        for (int32_t w = 0; w < 8; ++w) {
            abnormal |= warp[w];
        }
        flags[block] = abnormal;
    }
    asc_threadfence();
}

extern "C" __schedmode__(1) __global__ __aicore__ void spsv_soft64_wide_prefix(
    GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR ws, const SpsvTilingData t, uint32_t flagCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    SpsvBinary64ScratchView scratch{(__gm__ float*)(ws + t.yWideOffset), {static_cast<uint64_t>(t.m)}, flagCount};
    auto flags = scratch.admission;
    int32_t block = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (block == 0) {
        // The two admission arrays precede this per-solve prefix descriptor.
        *scratch.prefixLevels = 0;
    }
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(ws);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(state);
    if (SpsvComplexCertificateCompleted(ws, t.m) || !SpsvReplayFlagsRequireSolve(flags, flagCount)) {
        return;
    }
    auto levelPtr = (__gm__ const int32_t*)(ws + t.levelPtrOffset);
    auto levelRow = (__gm__ const int32_t*)(ws + t.levelRowOffset);
    int32_t prefix = ReplayWidePrefix(levelPtr, ((__gm__ const SpsvTilingData*)ws)->numLevels, t.m);
    if (prefix == 0) {
        return;
    }
    SpsvResolveComplexSolveCsr(rp, ci, values, ws, t);
    auto factors = scratch.factors;
    auto saved = scratch.saved;
    int32_t blocks = static_cast<int32_t>(AscendC::GetBlockNum());
    asc_vf_call<ReplayClearSaved>(dim3{256, 1, 1}, saved, t.m, block, blocks);
    SpsvTriplePrefixBarrier();
    for (int32_t level = 0; level < prefix; ++level) {
        asc_vf_call<ReplayPrefixLevel>(dim3{256, 1, 1}, (__gm__ const int32_t*)rp, (__gm__ const int32_t*)ci,
                                       (__gm__ const SpsvComplex64*)values, factors, saved, levelPtr, levelRow,
                                       level, (t.fillMode == 0) == (t.opA == 0), block, blocks);
        SpsvTriplePrefixBarrier();
    }
    asc_vf_call<ReplayPrefixRange>(dim3{256, 1, 1}, saved, levelRow, levelPtr[prefix], scratch.prefixRange,
                                     block, blocks);
    SpsvTriplePrefixBarrier();
    if (block == 0) {
        *scratch.prefixLevels = static_cast<uint32_t>(prefix);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(state);
    }
}

__aicore__ inline void ScalarLoadPrefix(__ubuf__ volatile uint64_t* local, __gm__ uint64_t* saved, int64_t m)
{
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)local, (__gm__ uint8_t*)saved, m * 2 * sizeof(uint64_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
}

__aicore__ inline void ScalarPublishSaved(__gm__ uint64_t* saved, __ubuf__ volatile uint64_t* local,
                                           int64_t m, int64_t completed, bool forward)
{
    // Keep the hot substitution in UB. Publish only completed rows, once,
    // rather than dirtying the scalar GM cache at every dependency step.
    int64_t first = forward ? 0 : m - completed;
    int64_t alignedFirst = first & ~int64_t{1};
    if (alignedFirst != first) {
        local[2 * alignedFirst] = 0;
        local[2 * alignedFirst + 1] = 0;
    }
    int64_t count = forward ? completed : m - alignedFirst;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(2);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(saved + 2 * alignedFirst),
                         (__ubuf__ uint8_t*)(local + 2 * alignedFirst), count * 2 * sizeof(uint64_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(2);
    saved[2 * m] = static_cast<uint64_t>(completed);
}


struct ScalarReplayData {
    __gm__ const int32_t* rowPtr;
    __gm__ const int32_t* columns;
    __gm__ const SpsvComplex64* values;
    __gm__ const int32_t* diagonal;
    __gm__ const uint64_t* factors;
    int64_t nnz;
    bool forward;
    bool nonUnit;
};

struct ScalarReplayTile {
    __ubuf__ volatile uint64_t* saved;
    __ubuf__ int32_t* columns;
    __ubuf__ float* values;
    __ubuf__ volatile uint64_t* terms;
    int32_t stagedBegin{-1};
};

template <bool PREPARED, bool NORMAL = false>
__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarAccumulateReplayTerms(
    const ScalarReplayTile& tile, int32_t offset, int32_t count, spsv_scalar::Complex sum)
{
    using namespace spsv_scalar;
    for (int32_t item = 0; item < count; ++item) {
        spsv_scalar::Complex term;
        if constexpr (PREPARED) {
            term = {tile.terms[2 * item], tile.terms[2 * item + 1]};
        } else {
            int32_t column = tile.columns[offset + item];
            SpsvComplex64 coefficient{tile.values[2 * (offset + item)], tile.values[2 * (offset + item) + 1]};
            if constexpr (NORMAL) {
                term = ScalarReplayNormalProduct(coefficient, {tile.saved[2 * column], tile.saved[2 * column + 1]});
            } else {
                term = ScalarReplayProduct(coefficient, {tile.saved[2 * column], tile.saved[2 * column + 1]});
            }
        }
        if constexpr (NORMAL) {
            sum = {AddReplayBounded(sum.re, Neg(term.re)), AddReplayBounded(sum.im, Neg(term.im))};
        } else {
            sum = {Sub(sum.re, term.re), Sub(sum.im, term.im)};
        }
    }
    return sum;
}

// Both complete event pairs are local to this unconditional batch
// boundary. Numerical fallbacks return only after the batch has completed.
__aicore__ inline void ScalarPrepareReplayTerms(const ScalarReplayTile& tile, int32_t offset,
                                              int32_t count, bool normal)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
    if (normal) {
        uint32_t threads = count <= 16 ? 32U : (count <= 32 ? 64U : (count <= 64 ? 128U : 256U));
        asc_vf_call<ReplayNormalProductChunk>(dim3{threads, 1, 1}, tile.columns, tile.values, tile.saved, tile.terms,
                                               offset, count);
    } else {
        asc_vf_call<ReplayProductChunk>(dim3{256, 1, 1}, tile.columns, tile.values, tile.saved, tile.terms,
                                         offset, count);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);
}

__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarReplayNumerator(
    const ScalarReplayData& data, ScalarReplayTile& tile, int64_t row, bool normal)
{
    spsv_scalar::Complex sum{data.factors[5 * row + 3], data.factors[5 * row + 4]};
    // Admission proves strict CSR order and the effective triangle. The
    // stored diagonal, when present, is therefore the first/last entry.
    // Exclude it once; every remaining term is a genuine predecessor.
    int32_t begin = data.rowPtr[row], end = data.rowPtr[row + 1];
    if (begin < end && data.columns[data.forward ? end - 1 : begin] == row) {
        if (data.forward) {
            --end;
        } else {
            ++begin;
        }
    }
    for (int32_t p = begin; p < end;) {
        if (tile.stagedBegin != p / 1024 * 1024) {
            tile.stagedBegin = ScalarStageEntries(data.columns, data.values, tile.columns, tile.values, p, data.nnz);
        }
        int32_t offset = p - tile.stagedBegin;
        int32_t count = end - p < 512 ? end - p : 512;
        count = count < 1024 - offset ? count : 1024 - offset;
        if (count <= (normal ? 4 : 8)) {
            sum = normal ? ScalarAccumulateReplayTerms<false, true>(tile, offset, count, sum) :
                           ScalarAccumulateReplayTerms<false>(tile, offset, count, sum);
        } else {
            ScalarPrepareReplayTerms(tile, offset, count, normal);
            sum = normal ? ScalarAccumulateReplayTerms<true, true>(tile, offset, count, sum) :
                           ScalarAccumulateReplayTerms<true>(tile, offset, count, sum);
        }
        p += count;
    }
    return sum;
}

__aicore__ __attribute__((always_inline)) inline spsv_scalar::Complex ScalarReplayFinishRow(
    const ScalarReplayData& data, int64_t row, spsv_scalar::Complex sum)
{
    if (!data.nonUnit) {
        return sum;
    }
    SpsvComplex64 diagonal{data.values[data.diagonal[row]].real, data.values[data.diagonal[row]].imag};
    return ScalarReplayDivide(sum, diagonal, data.factors, row);
}

__aicore__ inline bool ScalarReplayAdmission(GM_ADDR ws, const SpsvTilingData& t, uint32_t flagCount,
                                           uint32_t& suffixBegin, int32_t& prefix, bool& normal)
{
    AscendC::GlobalTensor<uint8_t> state;
    state.SetGlobalBuffer(ws);
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(state);
    SpsvBinary64ScratchView scratch{(__gm__ float*)(ws + t.yWideOffset), {static_cast<uint64_t>(t.m)}, flagCount};
    auto flags = scratch.admission;
    if (!SpsvReplayFlagsRequireSolve(flags, flagCount)) {
        return false;
    }
    suffixBegin = 0;
    normal = true;
    for (uint32_t block = 0; block < flagCount; ++block) {
        suffixBegin = suffixBegin > flags[flagCount + block] ? suffixBegin : flags[flagCount + block];
        normal &= (flags[block] & 4U) == 0U;
    }
    prefix = t.diagType != 0 && t.numLevels != t.m ? *scratch.prefixLevels : 0;
    if (prefix != 0) {
        for (uint32_t block = 0; block < flagCount; ++block) {
            normal &= scratch.prefixRange[block] == 0U;
        }
    }
    return true;
}

__aicore__ __attribute__((always_inline)) inline void ScalarStoreReplayResult(
    __ubuf__ volatile uint64_t* saved, int64_t row, spsv_scalar::Complex sum)
{
    saved[2 * row] = sum.re;
    saved[2 * row + 1] = sum.im;
}

extern "C" __global__ __aicore__ void spsv_scalar64_replay(
    GM_ADDR rp, GM_ADDR ci, GM_ADDR values, GM_ADDR rhs, GM_ADDR y, GM_ADDR ws,
    const SpsvTilingData tiling, uint32_t flagCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (SpsvComplexCertificateCompleted(ws, tiling.m)) {
        return;
    }
    uint32_t suffixBegin = 0;
    int32_t prefix = 0;
    bool normal = false;
    if (!ScalarReplayAdmission(ws, tiling, flagCount, suffixBegin, prefix, normal)) {
        return;
    }
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(tiling.m)};
    SpsvResolveComplexSolveCsr(rp, ci, values, ws, tiling);
    auto rowPtr = (__gm__ const int32_t*)rp;
    auto colInd = (__gm__ const int32_t*)ci;
    auto a = (__gm__ const SpsvComplex64*)values;
    auto diagonal = (__gm__ const int32_t*)(ws + tiling.diagPtrOffset);
    auto factors = (__gm__ const uint64_t*)(ws + tiling.yWideOffset + geometry.binary64FactorWord * sizeof(float));
    auto saved = (__gm__ uint64_t*)(ws + tiling.yWideOffset + geometry.binary64SavedWord * sizeof(float));
    __ubuf__ volatile uint64_t localSaved[2 * 12288];
    __ubuf__ int32_t localColumns[1024];
    __ubuf__ float localValues[2 * 1024];
    __ubuf__ volatile uint64_t terms[1024];
    using namespace spsv_scalar;
    bool forward = (tiling.fillMode == 0) == (tiling.opA == 0);
    ScalarReplayData data{rowPtr, colInd, a, diagonal, factors, tiling.nnz, forward, tiling.diagType == 0};
    ScalarReplayTile tile{localSaved, localColumns, localValues, terms};
    int64_t completed = tiling.m;
    auto levelPtr = (__gm__ const int32_t*)(ws + tiling.levelPtrOffset);
    auto levelRow = (__gm__ const int32_t*)(ws + tiling.levelRowOffset);
    int64_t first = prefix != 0 ? levelPtr[prefix] : 0;
    if (prefix != 0) {
        ScalarLoadPrefix(localSaved, saved, tiling.m);
    }
    for (int64_t step = first; step < tiling.m; ++step) {
        int64_t row = prefix != 0 ? levelRow[step] : (forward ? step : tiling.m - 1 - step);
        auto sum = ScalarReplayFinishRow(data, row, ScalarReplayNumerator(data, tile, row, normal));
        ScalarStoreReplayResult(localSaved, row, sum);
        normal &= spsv_scalar::ReplayOperandBounded(sum.re) && spsv_scalar::ReplayOperandBounded(sum.im);
        if (prefix == 0 && IsNaN(sum.re) && IsNaN(sum.im) && static_cast<uint64_t>(step + 1) >= suffixBegin) {
            completed = step + 1;
            break;
        }
    }
    ScalarPublishSaved(saved, localSaved, tiling.m, completed, forward);
    ((__gm__ SpsvTilingData*)ws)->primaryStatus = 2;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void ConvertSaved(__gm__ const uint64_t* saved,
                                                                       __gm__ SpsvComplex64* y, int64_t m, bool forward,
                                                                       int64_t start, int64_t stride)
{
    if (threadIdx.x == 0) {
        asc_dcci_entire((__gm__ void*)saved);
    }
    asc_syncthreads();
    uint64_t completed = saved[2 * m];
    for (int64_t row = start + threadIdx.x; row < m; row += stride) {
        int64_t step = forward ? row : m - 1 - row;
        spsv_soft::Complex result{spsv_soft::NaN, spsv_soft::NaN};
        if (static_cast<uint64_t>(step) < completed) {
            result = {saved[2 * row], saved[2 * row + 1]};
        }
        WriteSoft(y, row, result);
    }
    asc_threadfence();
}
extern "C" __global__ __aicore__ void spsv_soft64_convert(GM_ADDR y, GM_ADDR ws, const SpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (!SpsvBinary64PrimaryCompleted(ws, tiling.m)) {
        return;
    }
    const SpsvComplexWorkspaceGeometry geometry{static_cast<uint64_t>(tiling.m)};
    asc_vf_call<ConvertSaved>(
        dim3{256, 1, 1}, (__gm__ const uint64_t*)(ws + tiling.yWideOffset + geometry.binary64SavedWord * sizeof(float)),
        (__gm__ SpsvComplex64*)y, tiling.m, (tiling.fillMode == 0) == (tiling.opA == 0),
        static_cast<int64_t>(AscendC::GetBlockIdx()) * 256, static_cast<int64_t>(AscendC::GetBlockNum()) * 256);
}
void spsv_complex_binary64_primary_do(GM_ADDR rp, GM_ADDR ci, GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR ws,
                                      const SpsvTilingData& tiling, void* stream, bool early)
{
    uint32_t blocks = GetAivCoreCount();
    blocks = blocks > 0U ? blocks : tiling.numBlocks;
    spsv_replay_admission<<<blocks, nullptr, stream>>>(rp, ci, a, x, y, ws, tiling, early);
    spsv_soft64_prepare<<<blocks, nullptr, stream>>>(a, x, ws, tiling);
    if (tiling.m <= 12288) {
        if (tiling.diagType != 0 && tiling.numLevels != tiling.m) {
            spsv_soft64_wide_prefix<<<blocks, nullptr, stream>>>(rp, ci, a, ws, tiling, blocks);
        }
        spsv_scalar64_replay<<<1, nullptr, stream>>>(rp, ci, a, x, y, ws, tiling, blocks);
    } else {
        spsv_soft64_replay<<<1, nullptr, stream>>>(rp, ci, a, x, y, ws, tiling, blocks);
    }
    spsv_soft64_convert<<<blocks, nullptr, stream>>>(y, ws, tiling);
}
