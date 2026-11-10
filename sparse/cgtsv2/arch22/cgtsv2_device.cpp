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

#include "cgtsv2_device.h"

using namespace AscendC;

#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif

namespace {

using Cgtsv2::Complex;
using Cgtsv2::WideComplex;
using Cgtsv2::WideReal;

struct WideTensor {
    GlobalTensor<float> high;
    GlobalTensor<float> low;
    GlobalTensor<int64_t> exponent;

    __aicore__ inline void Bind(__gm__ uint8_t *base, uint64_t highOffset,
        uint64_t lowOffset, uint64_t exponentOffset, int32_t rows)
    {
        high.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + highOffset), 2LL * rows);
        low.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + lowOffset), 2LL * rows);
        exponent.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base + exponentOffset), 2LL * rows);
    }
};

__aicore__ inline WideComplex ReadWide(WideTensor &tensor, int32_t row)
{
    return {{tensor.high.GetValue(2LL * row), tensor.low.GetValue(2LL * row),
             tensor.exponent.GetValue(2LL * row)},
            {tensor.high.GetValue(2LL * row + 1), tensor.low.GetValue(2LL * row + 1),
             tensor.exponent.GetValue(2LL * row + 1)}};
}

__aicore__ inline void WriteWide(WideTensor &tensor, int32_t row, WideComplex value)
{
    tensor.high.SetValue(2LL * row, value.real.hi);
    tensor.high.SetValue(2LL * row + 1, value.imag.hi);
    tensor.low.SetValue(2LL * row, value.real.lo);
    tensor.low.SetValue(2LL * row + 1, value.imag.lo);
    tensor.exponent.SetValue(2LL * row, value.real.exponent);
    tensor.exponent.SetValue(2LL * row + 1, value.imag.exponent);
}

class Buffers {
public:
    __aicore__ inline void Init(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
        GM_ADDR workspace, const Cgtsv2TilingData &tiling)
    {
        m = tiling.m;
        n = tiling.n;
        ldb = tiling.ldb;
        parallelSingleRhs = n == 1 && tiling.blockNum > 1;
        lower.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dl), 2LL * m);
        diagonal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d), 2LL * m);
        upper.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(du), 2LL * m);
        rhs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B), 2ULL * ldb * n);
        auto bytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
        l.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bytes + tiling.lOffset), 2LL * (m - 1));
        dWork.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bytes + tiling.dOffset), 2LL * m);
        u.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bytes + tiling.uOffset), 2LL * (m - 1));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bytes + tiling.vOffset), 2LL * (m - 2));
        pivot.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(bytes + tiling.pivotOffset), m);
        status.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(bytes + tiling.statusOffset), 4);
        scratch.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bytes + tiling.rhsScratchOffset), 2LL * m);
        wideL.Bind(bytes, tiling.lOffset, tiling.lLowOffset, tiling.lExponentOffset, m - 1);
        wideD.Bind(bytes, tiling.dOffset, tiling.dLowOffset, tiling.dExponentOffset, m);
        wideU.Bind(bytes, tiling.uOffset, tiling.uLowOffset, tiling.uExponentOffset, m - 1);
        wideRhs.Bind(bytes, tiling.rhsScratchOffset, tiling.rhsLowOffset, tiling.rhsExponentOffset, m);
        correction.Bind(bytes, tiling.correctionHighOffset, tiling.correctionLowOffset,
            tiling.correctionExponentOffset, m);
    }

    int32_t m;
    int32_t n;
    int32_t ldb;
    bool directEligible = true;
    bool parallelSingleRhs = false;
    GlobalTensor<float> lower, diagonal, upper, rhs, l, dWork, u, v, scratch;
    GlobalTensor<uint32_t> pivot, status;
    WideTensor wideL, wideD, wideU, wideRhs, correction;
};

__aicore__ inline void Mark(Buffers &buffers, uint32_t flag, int32_t row)
{
    const uint32_t old = buffers.status.GetValue(0);
    buffers.status.SetValue(0, old | flag);
    const uint32_t index = flag == Cgtsv2::kSingular ? 1U : 2U;
    if (buffers.status.GetValue(index) == Cgtsv2::kNoRow) {
        buffers.status.SetValue(index, static_cast<uint32_t>(row));
    }
}

struct WideFactorResult {
    WideComplex multiplier;
    WideComplex diagonal;
    WideComplex upper;
    WideComplex nextDiagonal;
    WideComplex nextUpper;
    Complex secondUpper;
    uint32_t pivot;
};

__aicore__ inline WideFactorResult ExpandedFactorRow(Buffers &buffers, int32_t row,
    WideComplex a, WideComplex firstUpper, Complex rawLower, Complex rawDiagonal, Complex rawUpper)
{
    const WideComplex lower = Cgtsv2::Widen(rawLower);
    const WideComplex nextDiagonal = Cgtsv2::Widen(rawDiagonal);
    const WideComplex nextUpper = Cgtsv2::Widen(rawUpper);
    if (!Cgtsv2::Finite(a) || !Cgtsv2::Finite(firstUpper) || !Cgtsv2::Finite(lower) ||
        !Cgtsv2::Finite(nextDiagonal) || !Cgtsv2::Finite(nextUpper)) {
        Mark(buffers, Cgtsv2::kNonfinite, row);
    }
    WideFactorResult result{Cgtsv2::Widen(Complex{0.0f, 0.0f}), a, firstUpper,
        nextDiagonal, nextUpper, {0.0f, 0.0f}, static_cast<uint32_t>(row)};
    if (Cgtsv2::Zero(lower)) {
        if (Cgtsv2::Zero(a)) { Mark(buffers, Cgtsv2::kSingular, row); }
    } else if (!Cgtsv2::Finite(a) || !Cgtsv2::Finite(lower) ||
               Cgtsv2::NormGreaterEqualWide(a, lower)) {
        result.multiplier = Cgtsv2::DivideWide(lower, a);
        result.nextDiagonal = Cgtsv2::SubtractProductWide(nextDiagonal, result.multiplier, firstUpper);
    } else {
        result.multiplier = Cgtsv2::DivideWide(a, lower);
        result.diagonal = lower;
        result.upper = nextDiagonal;
        result.nextDiagonal = Cgtsv2::SubtractProductWide(firstUpper, result.multiplier, nextDiagonal);
        if (row < buffers.m - 2) {
            result.secondUpper = rawUpper;
            result.nextUpper = Cgtsv2::Negate(Cgtsv2::MultiplyWide(result.multiplier, nextUpper));
        }
        result.pivot = static_cast<uint32_t>(row + 1);
    }
    return result;
}

__aicore__ inline void FactorExpanded(Buffers &buffers)
{
    buffers.status.SetValue(0, Cgtsv2::kExpanded);
    WideComplex a = Cgtsv2::Widen(Cgtsv2::Read(buffers.diagonal, 0));
    WideComplex firstUpper = Cgtsv2::Widen(Cgtsv2::Read(buffers.upper, 0));
    for (int32_t row = 0; row < buffers.m - 1; ++row) {
        const Complex rawUpper = row < buffers.m - 2 ?
            Cgtsv2::Read(buffers.upper, row + 1) : Complex{0.0f, 0.0f};
        const WideFactorResult result = ExpandedFactorRow(buffers, row, a, firstUpper,
            Cgtsv2::Read(buffers.lower, row + 1), Cgtsv2::Read(buffers.diagonal, row + 1), rawUpper);
        WriteWide(buffers.wideL, row, result.multiplier);
        WriteWide(buffers.wideD, row, result.diagonal);
        WriteWide(buffers.wideU, row, result.upper);
        if (row < buffers.m - 2) { Cgtsv2::Write(buffers.v, row, result.secondUpper); }
        buffers.pivot.SetValue(row, result.pivot);
        a = result.nextDiagonal;
        firstUpper = result.nextUpper;
    }
    WriteWide(buffers.wideD, buffers.m - 1, a);
    buffers.pivot.SetValue(buffers.m - 1, static_cast<uint32_t>(buffers.m - 1));
    if (Cgtsv2::Zero(a)) { Mark(buffers, Cgtsv2::kSingular, buffers.m - 1); }
    if (!Cgtsv2::Finite(a)) { Mark(buffers, Cgtsv2::kNonfinite, buffers.m - 1); }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(buffers.dWork);
}

__aicore__ inline void SolveExpandedColumn(Buffers &buffers, WideTensor &rhs)
{
    for (int32_t row = 0; row < buffers.m - 1; ++row) {
        const WideComplex multiplier = ReadWide(buffers.wideL, row);
        const bool swapped = buffers.pivot.GetValue(row) != static_cast<uint32_t>(row);
        if (!swapped && Cgtsv2::Zero(multiplier)) { continue; }
        const WideComplex previous = ReadWide(rhs, row);
        const WideComplex current = ReadWide(rhs, row + 1);
        if (swapped) {
            WriteWide(rhs, row, current);
            WriteWide(rhs, row + 1, Cgtsv2::SubtractProductWide(previous, multiplier, current));
        } else {
            WriteWide(rhs, row + 1, Cgtsv2::SubtractProductWide(current, multiplier, previous));
        }
    }
    WideComplex next = Cgtsv2::DivideWide(ReadWide(rhs, buffers.m - 1),
        ReadWide(buffers.wideD, buffers.m - 1));
    WideComplex next2 = Cgtsv2::Widen(Complex{0.0f, 0.0f});
    WriteWide(rhs, buffers.m - 1, next);
    for (int32_t row = buffers.m - 2; row >= 0; --row) {
        WideComplex numerator;
        if (row < buffers.m - 2) {
            numerator = Cgtsv2::SubtractTwoProductsWide(ReadWide(rhs, row), ReadWide(buffers.wideU, row), next,
                Cgtsv2::Widen(Cgtsv2::Read(buffers.v, row)), next2);
        } else {
            numerator = Cgtsv2::SubtractProductWide(ReadWide(rhs, row), ReadWide(buffers.wideU, row), next);
        }
        const WideComplex answer = Cgtsv2::DivideWide(numerator, ReadWide(buffers.wideD, row));
        WriteWide(rhs, row, answer);
        next2 = next;
        next = answer;
    }
}

__aicore__ inline bool CopyExpandedOriginal(Buffers &buffers, int64_t offset)
{
    bool finite = true;
    for (int32_t row = 0; row < buffers.m; ++row) {
        const Complex original = Cgtsv2::Read(buffers.rhs, offset + row);
        finite = finite && Cgtsv2::Finite(original);
        WriteWide(buffers.wideRhs, row, Cgtsv2::Widen(original));
    }
    return finite;
}

__aicore__ inline bool BuildExpandedResidual(Buffers &buffers, int64_t offset, WideReal &maximum)
{
    bool finite = true;
    maximum = Cgtsv2::Widen(0.0f);
    const WideComplex zero = Cgtsv2::Widen(Complex{0.0f, 0.0f});
    for (int32_t row = 0; row < buffers.m; ++row) {
        const WideComplex previous = row == 0 ? zero : ReadWide(buffers.wideRhs, row - 1);
        const WideComplex current = ReadWide(buffers.wideRhs, row);
        const WideComplex next = row + 1 == buffers.m ? zero : ReadWide(buffers.wideRhs, row + 1);
        const Complex lower = row == 0 ? Complex{0.0f, 0.0f} : Cgtsv2::Read(buffers.lower, row);
        const Complex upper = row + 1 == buffers.m ? Complex{0.0f, 0.0f} : Cgtsv2::Read(buffers.upper, row);
        const WideComplex residual = Cgtsv2::ResidualWide(Cgtsv2::Read(buffers.rhs, offset + row),
            lower, Cgtsv2::Read(buffers.diagonal, row), upper, previous, current, next);
        finite = finite && Cgtsv2::Finite(residual);
        const auto squared = Cgtsv2::NormSquareWide(residual);
        if (Cgtsv2::AbsGreaterEqualWide(squared, maximum)) { maximum = squared; }
        WriteWide(buffers.correction, row, residual);
    }
    return finite;
}

__aicore__ inline bool ExpandedCorrectionBounded(Buffers &buffers)
{
    const auto one = Cgtsv2::Widen(1.0f);
    const auto toleranceSquared = Cgtsv2::Widen(1.0e-8f);
    for (int32_t row = 0; row < buffers.m; ++row) {
        const WideComplex current = ReadWide(buffers.wideRhs, row);
        const WideComplex correction = ReadWide(buffers.correction, row);
        if (!Cgtsv2::Finite(current) || !Cgtsv2::Finite(correction)) { return false; }
        auto magnitude = Cgtsv2::NormSquareWide(current);
        if (!Cgtsv2::AbsGreaterEqualWide(magnitude, one)) { magnitude = one; }
        if (!Cgtsv2::AbsGreaterEqualWide(Cgtsv2::MultiplyWide(magnitude, toleranceSquared),
            Cgtsv2::NormSquareWide(correction))) { return false; }
    }
    return true;
}

__aicore__ inline bool ExpandedCorrectionReducesResidual(Buffers &buffers,
    int64_t offset, WideReal before)
{
    auto after = Cgtsv2::Widen(0.0f);
    const WideComplex zero = Cgtsv2::Widen(Complex{0.0f, 0.0f});
    for (int32_t row = 0; row < buffers.m; ++row) {
        Complex coefficients[3] = {
            row == 0 ? Complex{0.0f, 0.0f} : Cgtsv2::Read(buffers.lower, row),
            Cgtsv2::Read(buffers.diagonal, row),
            row + 1 == buffers.m ? Complex{0.0f, 0.0f} : Cgtsv2::Read(buffers.upper, row)};
        WideComplex current[3];
        WideComplex correction[3];
        for (int32_t i = 0; i < 3; ++i) {
            const int32_t neighbor = row + i - 1;
            const bool active = neighbor >= 0 && neighbor < buffers.m;
            current[i] = active ? ReadWide(buffers.wideRhs, neighbor) : zero;
            correction[i] = active ? ReadWide(buffers.correction, neighbor) : zero;
            if (!Cgtsv2::Finite(current[i]) || !Cgtsv2::Finite(correction[i])) { return false; }
        }
        const auto residual = Cgtsv2::CorrectedResidualWide(
            Cgtsv2::Read(buffers.rhs, offset + row), coefficients, current, correction);
        if (!Cgtsv2::Finite(residual)) { return false; }
        const auto squared = Cgtsv2::NormSquareWide(residual);
        if (Cgtsv2::AbsGreaterEqualWide(squared, after)) { after = squared; }
    }
    return !Cgtsv2::AbsGreaterEqualWide(after, before);
}

__aicore__ inline void SolveExpanded(Buffers &buffers, uint32_t flags)
{
    // All extended-range arrays describe one shared RHS. Other launch blocks
    // return before this routine; original B remains available until narrowing.
    for (int64_t column = 0; column < buffers.n; ++column) {
        const int64_t offset = column * buffers.ldb;
        const bool finiteRhs = CopyExpandedOriginal(buffers, offset);
        SolveExpandedColumn(buffers, buffers.wideRhs);
        bool apply = false;
        WideReal maximumResidual = Cgtsv2::Widen(0.0f);
        if (finiteRhs && (flags & (Cgtsv2::kSingular | Cgtsv2::kNonfinite)) == 0 &&
            BuildExpandedResidual(buffers, offset, maximumResidual)) {
            SolveExpandedColumn(buffers, buffers.correction);
            apply = ExpandedCorrectionBounded(buffers) ||
                ExpandedCorrectionReducesResidual(buffers, offset, maximumResidual);
        }
        for (int32_t row = 0; row < buffers.m; ++row) {
            WideComplex answer = ReadWide(buffers.wideRhs, row);
            if (apply) { answer = Cgtsv2::AddWide(answer, ReadWide(buffers.correction, row)); }
            Cgtsv2::Write(buffers.rhs, offset + row, Cgtsv2::NarrowWide(answer));
        }
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(buffers.rhs);
}

struct FactorResult {
    Complex multiplier;
    Complex diagonal;
    Complex upper;
    Complex secondUpper;
    Complex nextDiagonal;
    Complex nextUpper;
    uint32_t pivot;
};

__aicore__ inline FactorResult ComputeFactorRow(Buffers &buffers, int32_t row, Complex a,
    Complex firstUpper, Complex lower, Complex nextDiagonal, Complex nextUpper)
{
    if (!Cgtsv2::Finite(a) || !Cgtsv2::Finite(lower) || !Cgtsv2::Finite(firstUpper) ||
        !Cgtsv2::Finite(nextDiagonal) || !Cgtsv2::Finite(nextUpper)) {
        Mark(buffers, Cgtsv2::kNonfinite, row);
    }
    FactorResult result{{0.0f, 0.0f}, a, firstUpper, {0.0f, 0.0f},
                        nextDiagonal, nextUpper, static_cast<uint32_t>(row)};
    if (Cgtsv2::Zero(lower)) {
        // ZGTSV skips elimination for a structural zero subdiagonal.
        if (Cgtsv2::Zero(a)) { Mark(buffers, Cgtsv2::kSingular, row); }
    } else if (!Cgtsv2::Finite(a) || !Cgtsv2::Finite(lower) ||
               Cgtsv2::NormGreaterEqualDispatch(a, lower, buffers.directEligible)) {
        result.multiplier = Cgtsv2::DivideDispatch(lower, a, buffers.directEligible);
        result.nextDiagonal = Cgtsv2::SubtractProductDispatch(
            nextDiagonal, result.multiplier, firstUpper, buffers.directEligible);
    } else {
        result.multiplier = Cgtsv2::DivideDispatch(a, lower, buffers.directEligible);
        result.diagonal = lower;
        result.upper = nextDiagonal;
        result.nextDiagonal = Cgtsv2::SubtractProductDispatch(
            firstUpper, result.multiplier, nextDiagonal, buffers.directEligible);
        if (row < buffers.m - 2) {
            result.secondUpper = nextUpper;
            result.nextUpper = Cgtsv2::Negate(Cgtsv2::Multiply(result.multiplier, nextUpper));
        }
        result.pivot = static_cast<uint32_t>(row + 1);
    }
    if (!Cgtsv2::Finite(result.multiplier) || !Cgtsv2::Finite(result.nextDiagonal)) {
        Mark(buffers, Cgtsv2::kNonfinite, row);
    }
    return result;
}

template <typename FloatTensor, typename PivotTensor>
__aicore__ inline void WriteFactors(FloatTensor &l, FloatTensor &d, FloatTensor &u,
    FloatTensor &v, PivotTensor &pivot, int32_t row, bool hasSecondUpper, const FactorResult &result)
{
    Cgtsv2::Write(l, row, result.multiplier);
    Cgtsv2::Write(d, row, result.diagonal);
    Cgtsv2::Write(u, row, result.upper);
    if (hasSecondUpper) { Cgtsv2::Write(v, row, result.secondUpper); }
    pivot.SetValue(row, result.pivot);
}

__aicore__ inline void InitializeStatus(Buffers &buffers)
{
    // Same-address input/workspace reuse must not retain previous DCache data.
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(buffers.diagonal);
    buffers.status.SetValue(0, 0);
    buffers.status.SetValue(1, Cgtsv2::kNoRow);
    buffers.status.SetValue(2, Cgtsv2::kNoRow);
    buffers.status.SetValue(3, 0);
    buffers.directEligible = true;
}

__aicore__ inline void FinalStatus(Buffers &buffers, Complex lastDiagonal)
{
    if (Cgtsv2::Zero(lastDiagonal)) { Mark(buffers, Cgtsv2::kSingular, buffers.m - 1); }
    if (!Cgtsv2::Finite(lastDiagonal)) { Mark(buffers, Cgtsv2::kNonfinite, buffers.m - 1); }
    if (buffers.directEligible && buffers.status.GetValue(0) == 0) {
        buffers.status.SetValue(0, Cgtsv2::kDirectEligible);
    }
}

__aicore__ inline void FactorRow(Buffers &buffers, int32_t row, Complex &a, Complex &firstUpper)
{
    const Complex lower = Cgtsv2::Read(buffers.lower, row + 1);
    const Complex nextDiagonal = Cgtsv2::Read(buffers.diagonal, row + 1);
    Complex nextUpper{0.0f, 0.0f};
    if (row < buffers.m - 2) { nextUpper = Cgtsv2::Read(buffers.upper, row + 1); }
    const FactorResult result = ComputeFactorRow(buffers, row, a, firstUpper, lower, nextDiagonal, nextUpper);
    WriteFactors(buffers.l, buffers.dWork, buffers.u, buffers.v, buffers.pivot,
        row, row < buffers.m - 2, result);
    a = result.nextDiagonal;
    firstUpper = result.nextUpper;
}

__aicore__ inline bool CertifyDirectGm(Buffers &buffers)
{
    for (int32_t row = 0; row < buffers.m; ++row) {
        const Complex lower = row != 0 ? Cgtsv2::Read(buffers.lower, row) : Complex{0.0f, 0.0f};
        const Complex upper = row + 1 < buffers.m ? Cgtsv2::Read(buffers.upper, row) : Complex{0.0f, 0.0f};
        if (!Cgtsv2::DirectRowEligible(lower, Cgtsv2::Read(buffers.diagonal, row), upper)) { return false; }
    }
    return true;
}

__aicore__ inline bool DirectRhsValueEligible(Complex value)
{
    // Keep ordinary FP32 solves in the bounded-RHS regime used by Jacobi.
    // Larger finite RHS can expose small complex cross terms after cancellation
    // even when the final solution is modest; use expanded-range solving there.
    constexpr float limit = 16.0f;
    return Cgtsv2::Finite(value) && Cgtsv2::MaximumComponent(value) <=
        Cgtsv2::MagnitudeBits(limit);
}

__aicore__ inline bool CertifyDirectRhsGm(Buffers &buffers)
{
    for (int64_t column = 0; column < buffers.n; ++column) {
        for (int32_t row = 0; row < buffers.m; ++row) {
            if (!DirectRhsValueEligible(Cgtsv2::Read(buffers.rhs, column * buffers.ldb + row))) {
                return false;
            }
        }
    }
    return true;
}

__aicore__ inline void Factor(Buffers &buffers)
{
    InitializeStatus(buffers);
    buffers.directEligible = CertifyDirectGm(buffers) && CertifyDirectRhsGm(buffers);
    if (!buffers.directEligible) { FactorExpanded(buffers); return; }
    Complex a = Cgtsv2::Read(buffers.diagonal, 0);
    Complex firstUpper = Cgtsv2::Read(buffers.upper, 0);
    for (int32_t row = 0; row < buffers.m - 1; ++row) {
        FactorRow(buffers, row, a, firstUpper);
    }
    Cgtsv2::Write(buffers.dWork, buffers.m - 1, a);
    buffers.pivot.SetValue(buffers.m - 1, static_cast<uint32_t>(buffers.m - 1));
    FinalStatus(buffers, a);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(buffers.dWork);
}

template <typename FloatTensor, typename PivotTensor, typename RhsTensor>
__aicore__ inline void SolveColumn(int32_t m, FloatTensor &l, FloatTensor &d,
    FloatTensor &u, FloatTensor &v, PivotTensor &pivot, RhsTensor &rhs, int64_t offset, bool fast = false)
{
    for (int32_t row = 0; row < m - 1; ++row) {
        const Complex multiplier = Cgtsv2::Read(l, row);
        const bool swapped = pivot.GetValue(row) != static_cast<uint32_t>(row);
        if (!swapped && Cgtsv2::Zero(multiplier)) {
            // This is the structural no-elimination branch of ZGTSV, not a
            // general permission to skip zero products during back substitution.
            continue;
        }
        const Complex previous = Cgtsv2::Read(rhs, offset + row);
        const Complex current = Cgtsv2::Read(rhs, offset + row + 1);
        if (swapped) {
            Cgtsv2::Write(rhs, offset + row, current);
            Cgtsv2::Write(rhs, offset + row + 1,
                Cgtsv2::SubtractProductDispatch(previous, multiplier, current, fast));
        } else {
            Cgtsv2::Write(rhs, offset + row + 1,
                Cgtsv2::SubtractProductDispatch(current, multiplier, previous, fast));
        }
    }
    Complex next = Cgtsv2::DivideDispatch(Cgtsv2::Read(rhs, offset + m - 1), Cgtsv2::Read(d, m - 1), fast);
    Complex next2{0.0f, 0.0f};
    Cgtsv2::Write(rhs, offset + m - 1, next);
    for (int32_t row = m - 2; row >= 0; --row) {
        Complex numerator = Cgtsv2::SubtractProductDispatch(Cgtsv2::Read(rhs, offset + row),
            Cgtsv2::Read(u, row), next, fast);
        if (row < m - 2) {
            numerator = Cgtsv2::SubtractProductDispatch(numerator, Cgtsv2::Read(v, row), next2, fast);
        }
        const Complex answer = Cgtsv2::DivideDispatch(numerator, Cgtsv2::Read(d, row), fast);
        Cgtsv2::Write(rhs, offset + row, answer);
        next2 = next;
        next = answer;
    }
}

__aicore__ inline DataCopyExtParams OneCopy(uint32_t bytes)
{
    return DataCopyExtParams{1, bytes, 0, 0, 0};
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> NoPad()
{
    return DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)};
}

constexpr uint32_t kJacobiRows = kCgtsv2JacobiCenterRows + 2U * kCgtsv2JacobiIters;
constexpr uint32_t kJacobiGuard = 8U;

struct JacobiEvents {
    event_t mteToV, vToMte, vToS, sToV, vToStore, storeToV;

    __aicore__ explicit JacobiEvents(TPipe &pipe)
    {
        mteToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        vToMte = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        vToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        sToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
        vToStore = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        storeToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void BeforeLoad()
    {
        SetFlag<HardEvent::V_MTE2>(vToMte);
        WaitFlag<HardEvent::V_MTE2>(vToMte);
    }

    __aicore__ inline void Loaded()
    {
        SetFlag<HardEvent::MTE2_V>(mteToV);
        WaitFlag<HardEvent::MTE2_V>(mteToV);
    }
};

__aicore__ inline LocalTensor<float> JacobiArray(LocalTensor<float> storage, uint32_t index)
{
    return storage[index * kCgtsv2JacobiStride];
}

__aicore__ inline void JacobiIndices(LocalTensor<float> storage, uint32_t shift, uint32_t count)
{
    auto indices = JacobiArray(storage, 14).ReinterpretCast<uint32_t>();
    CreateVecIndex(indices.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(indices, indices, shift, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiLoadPair(GlobalTensor<float> source, int64_t first,
    uint32_t count, uint32_t pad, LocalTensor<float> storage, uint32_t realIndex,
    uint32_t imagIndex, JacobiEvents &events)
{
    auto raw = JacobiArray(storage, 15);
    auto indices = JacobiArray(storage, 14).ReinterpretCast<uint32_t>();
    events.BeforeLoad();
    DataCopyPad(raw, source[2LL * first], OneCopy(8U * count), NoPad<float>());
    events.Loaded();
    Gather(JacobiArray(storage, realIndex)[pad], raw, indices, 0U, count);
    Gather(JacobiArray(storage, imagIndex)[pad], raw, indices, 4U, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiEndpointZeros(Buffers &buffers, int32_t first,
    uint32_t count, uint32_t pad, LocalTensor<float> storage, JacobiEvents &events)
{
    SetFlag<HardEvent::V_S>(events.vToS);
    WaitFlag<HardEvent::V_S>(events.vToS);
    if (first == 0) {
        JacobiArray(storage, 0).SetValue(pad, 0.0f);
        JacobiArray(storage, 1).SetValue(pad, 0.0f);
    }
    if (first + static_cast<int32_t>(count) == buffers.m) {
        JacobiArray(storage, 2).SetValue(pad + count - 1U, 0.0f);
        JacobiArray(storage, 3).SetValue(pad + count - 1U, 0.0f);
    }
    SetFlag<HardEvent::S_V>(events.sToV);
    WaitFlag<HardEvent::S_V>(events.sToV);
}

__aicore__ inline void JacobiLoadCoefficients(Buffers &buffers, int32_t first,
    uint32_t count, uint32_t pad, LocalTensor<float> storage, JacobiEvents &events)
{
    JacobiIndices(storage, 3U, count);
    JacobiLoadPair(buffers.lower, first, count, pad, storage, 0, 1, events);
    JacobiLoadPair(buffers.upper, first, count, pad, storage, 2, 3, events);
    JacobiLoadPair(buffers.diagonal, first, count, pad, storage, 10, 11, events);
    JacobiEndpointZeros(buffers, first, count, pad, storage, events);
}

__aicore__ inline void JacobiOffBound(LocalTensor<float> high, LocalTensor<float> low,
    LocalTensor<float> real, LocalTensor<float> imag, LocalTensor<float> storage, uint32_t count)
{
    auto largest = JacobiArray(storage, 17), product = JacobiArray(storage, 18);
    auto candidate = JacobiArray(storage, 19);
    Max(largest, real, imag, count);
    Min(low, real, imag, count); PipeBarrier<PIPE_V>();
    Duplicate(high, 0.0f, count); PipeBarrier<PIPE_V>();
    // Chords of sqrt(1+t*t), t=min/max, on four intervals of width 1/4.
    // Convexity makes their maximum an upper bound. Constants round upward;
    // inflation covers the FP32 products/addition, including tiny FTZ tails.
    constexpr float intercepts[] = {0x1.000002p+0f, 0x1.e314e8p-1f, 0x1.b54cdcp-1f, 0x1.83c49cp-1f};
    constexpr float slopes[] = {0x1.f83d9cp-4f, 0x1.656836p-2f, 0x1.0e4434p-1f, 0x1.504f36p-1f};
    for (uint32_t segment = 0; segment < 4U; ++segment) {
        Muls(product, largest, intercepts[segment], count);
        Muls(candidate, low, slopes[segment], count); PipeBarrier<PIPE_V>();
        Add(candidate, product, candidate, count); PipeBarrier<PIPE_V>();
        Max(high, high, candidate, count); PipeBarrier<PIPE_V>();
    }
    Muls(high, high, 1.0f + 0x1p-18f, count); PipeBarrier<PIPE_V>();
    Adds(high, high, 0x1p-100f, count); PipeBarrier<PIPE_V>();
}

__aicore__ inline bool JacobiCheckCoefficients(LocalTensor<float> storage,
    uint32_t count, JacobiEvents &events)
{
    auto lr = JacobiArray(storage, 0), li = JacobiArray(storage, 1);
    auto ur = JacobiArray(storage, 2), ui = JacobiArray(storage, 3);
    auto result = JacobiArray(storage, 4), scratch = JacobiArray(storage, 5);
    auto lb = JacobiArray(storage, 6), lo = JacobiArray(storage, 7);
    auto ub = JacobiArray(storage, 8), uo = JacobiArray(storage, 9);
    auto dr = JacobiArray(storage, 10), di = JacobiArray(storage, 11);
    auto maximum = JacobiArray(storage, 12), sum = JacobiArray(storage, 13);
    Abs(lr, lr, count); Abs(li, li, count); Abs(ur, ur, count); Abs(ui, ui, count);
    Abs(dr, dr, count); Abs(di, di, count);
    PipeBarrier<PIPE_V>();
    Add(sum, lr, li, count); Max(maximum, dr, di, count);
    PipeBarrier<PIPE_V>();
    Add(sum, sum, ur, count); PipeBarrier<PIPE_V>();
    Add(sum, sum, ui, count); PipeBarrier<PIPE_V>();
    Add(sum, sum, dr, count); PipeBarrier<PIPE_V>();
    Add(sum, sum, di, count); PipeBarrier<PIPE_V>();
    JacobiOffBound(lb, lo, lr, li, storage, count);
    JacobiOffBound(ub, uo, ur, ui, storage, count);
    Add(lb, lb, ub, count); PipeBarrier<PIPE_V>();
    Muls(lb, lb, kCgtsv2JacobiDominance, count); PipeBarrier<PIPE_V>();
    Sub(lb, maximum, lb, count); PipeBarrier<PIPE_V>();
    ReduceSum(result, sum, scratch, count); PipeBarrier<PIPE_V>();
    ReduceMin(result[8], lb, scratch, count, false); PipeBarrier<PIPE_V>();
    ReduceMin(result[16], maximum, scratch, count, false); PipeBarrier<PIPE_V>();
    ReduceMax(result[24], maximum, scratch, count, false);
    SetFlag<HardEvent::V_S>(events.vToS);
    WaitFlag<HardEvent::V_S>(events.vToS);
    return Cgtsv2::Finite(result.GetValue(0)) && result.GetValue(8) > 0.0f &&
        result.GetValue(16) >= 1.0f && result.GetValue(24) <= 8.0f;
}

__aicore__ inline void JacobiPreserveRhs(Buffers &buffers, int32_t first, uint32_t count,
    LocalTensor<float> raw, JacobiEvents &events)
{
    if (!buffers.parallelSingleRhs) { return; }
    // The factor/preparation kernel owns the complete original-B copy. Publish
    // status16 only after every DMA finishes, before the same-stream solve.
    SetFlag<HardEvent::V_MTE3>(events.vToStore);
    WaitFlag<HardEvent::V_MTE3>(events.vToStore);
    DataCopyPad(buffers.scratch[2LL * first], raw, OneCopy(8U * count));
    SetFlag<HardEvent::MTE3_V>(events.storeToV);
    WaitFlag<HardEvent::MTE3_V>(events.storeToV);
}

__aicore__ inline bool JacobiCheckRhs(Buffers &buffers, LocalTensor<float> storage,
    JacobiEvents &events)
{
    constexpr uint32_t checkRows = 2048U;
    constexpr uint32_t checkFloats = 2U * checkRows;
    constexpr uint32_t resultOffset = 2U * checkFloats;
    constexpr uint32_t scratchOffset = resultOffset + 16U;
    static_assert(checkFloats % 8U == 0 && scratchOffset % 8U == 0,
        "RHS preflight buffers must start on 32-byte boundaries");
    static_assert((scratchOffset + checkRows) * sizeof(float) <= kCgtsv2JacobiStorageBytes,
        "RHS preflight must fit the existing Jacobi storage");
    // Coefficient certification is complete; solve/fallback reload their data.
    auto raw = storage, absolute = storage[checkFloats];
    auto result = storage[resultOffset], scratch = storage[scratchOffset];
    for (int64_t column = 0; column < buffers.n; ++column) {
        for (int32_t first = 0; first < buffers.m;) {
            const uint32_t count = static_cast<uint32_t>(buffers.m - first) < checkRows ?
                static_cast<uint32_t>(buffers.m - first) : checkRows;
            events.BeforeLoad();
            DataCopyPad(raw, buffers.rhs[2LL * (column * buffers.ldb + first)],
                OneCopy(8U * count), NoPad<float>());
            events.Loaded();
            Abs(absolute, raw, 2U * count); PipeBarrier<PIPE_V>();
            ReduceSum(result, absolute, scratch, 2U * count); PipeBarrier<PIPE_V>();
            ReduceMax(result[8], absolute, scratch, 2U * count, false);
            SetFlag<HardEvent::V_S>(events.vToS);
            WaitFlag<HardEvent::V_S>(events.vToS);
            if (!Cgtsv2::Finite(result.GetValue(0)) || !(result.GetValue(8) <= 16.0f)) { return false; }
            JacobiPreserveRhs(buffers, first, count, raw, events);
            first += static_cast<int32_t>(count);
        }
    }
    return true;
}

__aicore__ inline bool TryCertifyJacobi(Buffers &buffers, LocalTensor<float> storage, TPipe &pipe)
{
    if (buffers.m < kCgtsv2JacobiMinRows) { return false; }
    // The conservative off-norm bound and diagonal lower bound certify
    // ||D^-1(L+U)||_inf < 0.74. Full matrix/RHS preflight precedes any B store.
    JacobiEvents events(pipe);
    for (int32_t first = 0; first < buffers.m;) {
        const uint32_t count = static_cast<uint32_t>(buffers.m - first) < kJacobiRows ?
            static_cast<uint32_t>(buffers.m - first) : kJacobiRows;
        JacobiLoadCoefficients(buffers, first, count, 0U, storage, events);
        if (!JacobiCheckCoefficients(storage, count, events)) { return false; }
        first += static_cast<int32_t>(count);
    }
    return JacobiCheckRhs(buffers, storage, events);
}

__aicore__ inline void JacobiNormalize(LocalTensor<float> real, LocalTensor<float> imag,
    LocalTensor<float> storage)
{
    auto dr = JacobiArray(storage, 10), di = JacobiArray(storage, 11);
    auto inverse = JacobiArray(storage, 12), product = JacobiArray(storage, 13);
    auto temporary = JacobiArray(storage, 6), second = JacobiArray(storage, 7);
    Mul(product, real, dr, kJacobiRows);
    Mul(temporary, imag, di, kJacobiRows);
    PipeBarrier<PIPE_V>();
    Add(product, product, temporary, kJacobiRows);
    Mul(temporary, imag, dr, kJacobiRows);
    Mul(second, real, di, kJacobiRows);
    PipeBarrier<PIPE_V>();
    Sub(temporary, temporary, second, kJacobiRows); PipeBarrier<PIPE_V>();
    Mul(real, product, inverse, kJacobiRows);
    Mul(imag, temporary, inverse, kJacobiRows); PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiLoadOriginalRhs(Buffers &buffers, int64_t offset,
    int32_t center, int32_t first, uint32_t count, uint32_t centerCount,
    uint32_t pad, LocalTensor<float> storage, JacobiEvents &events)
{
    if (buffers.parallelSingleRhs) {
        // All tiles read the completed original copy. B writes by another core
        // cannot affect the halo, so no private left-ring repair is needed.
        JacobiLoadPair(buffers.scratch, first, count, pad, storage, 4, 5, events);
        return;
    }
    auto raw = JacobiArray(storage, 15);
    auto ring = storage[kCgtsv2JacobiArrayCount * kCgtsv2JacobiStride];
    auto indices = JacobiArray(storage, 14).ReinterpretCast<uint32_t>();
    events.BeforeLoad();
    DataCopyPad(raw, buffers.rhs[2LL * (offset + first)], OneCopy(8U * count), NoPad<float>());
    events.Loaded();
    // A previous tile has overwritten exactly this tile's left halo. Restore
    // its original B from private UB; no shared GM column scratch is involved.
    if (center != 0) {
        DataCopy(raw, ring, 2U * kCgtsv2JacobiIters);
        PipeBarrier<PIPE_V>();
    }
    if (center + static_cast<int32_t>(centerCount) < buffers.m) {
        const uint32_t ringFirst = static_cast<uint32_t>(center - first) + centerCount - kCgtsv2JacobiIters;
        DataCopy(ring, raw[2U * ringFirst], 2U * kCgtsv2JacobiIters);
        PipeBarrier<PIPE_V>();
    }
    Gather(JacobiArray(storage, 4)[pad], raw, indices, 0U, count);
    Gather(JacobiArray(storage, 5)[pad], raw, indices, 4U, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiInverseNewtonStep(LocalTensor<float> inverse,
    LocalTensor<float> denominator, LocalTensor<float> correction)
{
    Mul(correction, denominator, inverse, kJacobiRows); PipeBarrier<PIPE_V>();
    Muls(correction, correction, -1.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    Adds(correction, correction, 2.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    Mul(inverse, inverse, correction, kJacobiRows); PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiAccurateInverse(LocalTensor<float> storage)
{
    auto inverse = JacobiArray(storage, 12), denominator = JacobiArray(storage, 13);
    auto correction = JacobiArray(storage, 6);
    // DAV_C220 Reciprocal is an approximation. Preserve |D|^2 and refine the
    // seed twice before it normalizes A or B. Certification bounds |D|^2 away
    // from zero, so these Newton steps stay in the normal finite FP32 range.
    Adds(denominator, inverse, 0.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    Reciprocal(inverse, inverse, kJacobiRows); PipeBarrier<PIPE_V>();
    JacobiInverseNewtonStep(inverse, denominator, correction);
    JacobiInverseNewtonStep(inverse, denominator, correction);
}

// Bounded FP32 error-free transformations; no single-rounding FMA assumption.
enum JacobiRefinementSlot : uint32_t {
    kJacobiRawLowerReal = 17, kJacobiRawLowerImag, kJacobiRawUpperReal, kJacobiRawUpperImag,
    kJacobiRawDiagonalReal, kJacobiRawDiagonalImag, kJacobiRawRhsReal, kJacobiRawRhsImag,
    kJacobiInitialReal, kJacobiInitialImag, kJacobiNegativeLowerImag, kJacobiNegativeUpperImag,
    kJacobiResidualRealHigh, kJacobiResidualRealLow, kJacobiResidualImagHigh, kJacobiResidualImagLow,
    kJacobiProduct, kJacobiProductError, kJacobiCoefficientRealHigh, kJacobiCoefficientRealLow,
    kJacobiCoefficientImagHigh, kJacobiCoefficientImagLow, kJacobiValueRealHigh, kJacobiValueRealLow,
    kJacobiValueImagHigh, kJacobiValueImagLow, kJacobiSum, kJacobiRound0, kJacobiRound1,
    kJacobiInverseCache
};
static_assert(kJacobiInverseCache + 1U == kCgtsv2JacobiArrayCount, "Jacobi UB slots must match Host allocation");

__aicore__ inline void JacobiAdd(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b)
{
    const BinaryRepeatParams repeat{1, 1, 1, 8, 8, 8};
    Add<float, false>(dst, a, b, MASK_PLACEHOLDER, 1, repeat);
}

__aicore__ inline void JacobiSubtract(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b)
{
    const BinaryRepeatParams repeat{1, 1, 1, 8, 8, 8};
    Sub<float, false>(dst, a, b, MASK_PLACEHOLDER, 1, repeat);
}

__aicore__ inline void JacobiMultiply(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b)
{
    const BinaryRepeatParams repeat{1, 1, 1, 8, 8, 8};
    Mul<float, false>(dst, a, b, MASK_PLACEHOLDER, 1, repeat);
}

__aicore__ inline void JacobiScale(LocalTensor<float> dst, LocalTensor<float> a, float scale)
{
    const UnaryRepeatParams repeat{1, 1, 8, 8};
    Muls<float, false>(dst, a, scale, MASK_PLACEHOLDER, 1, repeat);
}

__aicore__ inline void JacobiMultiplyAdd(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b)
{
    const BinaryRepeatParams repeat{1, 1, 1, 8, 8, 8};
    MulAddDst<float, float, false>(dst, a, b, MASK_PLACEHOLDER, 1, repeat);
}

struct JacobiVectorParts {
    LocalTensor<float> raw, high, low;
};
struct JacobiComplexParts {
    JacobiVectorParts real, imag;
};
struct JacobiVectorPair {
    LocalTensor<float> high, low;
};
struct JacobiResidualPair {
    JacobiVectorPair real, imag;
};
struct JacobiProductWork {
    LocalTensor<float> product, error, sum, round0, round1;
};
struct JacobiComplexView {
    LocalTensor<float> real, imag;
};

__aicore__ inline JacobiComplexParts JacobiParts(LocalTensor<float> storage,
    JacobiComplexView value, uint32_t first)
{
    return {{value.real, JacobiArray(storage, first), JacobiArray(storage, first + 1U)},
        {value.imag, JacobiArray(storage, first + 2U), JacobiArray(storage, first + 3U)}};
}

__aicore__ inline void JacobiSplitPair(JacobiComplexParts &a, JacobiComplexParts &b)
{
    // Admission bounds coefficients and iterates; multiplying by 2^12+1 cannot overflow.
    JacobiScale(a.real.high, a.real.raw, 4097.0f);
    JacobiScale(a.imag.high, a.imag.raw, 4097.0f);
    JacobiScale(b.real.high, b.real.raw, 4097.0f);
    JacobiScale(b.imag.high, b.imag.raw, 4097.0f); PipeBarrier<PIPE_V>();
    JacobiSubtract(a.real.low, a.real.high, a.real.raw);
    JacobiSubtract(a.imag.low, a.imag.high, a.imag.raw);
    JacobiSubtract(b.real.low, b.real.high, b.real.raw);
    JacobiSubtract(b.imag.low, b.imag.high, b.imag.raw); PipeBarrier<PIPE_V>();
    JacobiSubtract(a.real.high, a.real.high, a.real.low);
    JacobiSubtract(a.imag.high, a.imag.high, a.imag.low);
    JacobiSubtract(b.real.high, b.real.high, b.real.low);
    JacobiSubtract(b.imag.high, b.imag.high, b.imag.low); PipeBarrier<PIPE_V>();
    JacobiSubtract(a.real.low, a.real.raw, a.real.high);
    JacobiSubtract(a.imag.low, a.imag.raw, a.imag.high);
    JacobiSubtract(b.real.low, b.real.raw, b.real.high);
    JacobiSubtract(b.imag.low, b.imag.raw, b.imag.high); PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiErrorProduct(const JacobiVectorParts &a,
    const JacobiVectorParts &b, JacobiProductWork &work)
{
    JacobiMultiply(work.product, a.raw, b.raw);
    JacobiMultiply(work.error, a.high, b.high); PipeBarrier<PIPE_V>();
    JacobiSubtract(work.error, work.error, work.product); PipeBarrier<PIPE_V>();
    // Split operands have at most 12 significant bits. Their products are exact
    // in FP32 (apart from negligible underflow), so fused/separate adds agree.
    JacobiMultiplyAdd(work.error, a.high, b.low); PipeBarrier<PIPE_V>();
    JacobiMultiplyAdd(work.error, a.low, b.high); PipeBarrier<PIPE_V>();
    JacobiMultiplyAdd(work.error, a.low, b.low); PipeBarrier<PIPE_V>();
}

template <bool subtract>
__aicore__ inline void JacobiAccumulateProduct(JacobiVectorPair &accumulator,
    const JacobiVectorParts &a, const JacobiVectorParts &b, JacobiProductWork &work)
{
    JacobiErrorProduct(a, b, work);
    if constexpr (subtract) {
        JacobiSubtract(work.sum, accumulator.high, work.product); PipeBarrier<PIPE_V>();
        JacobiSubtract(work.round0, accumulator.high, work.sum); PipeBarrier<PIPE_V>();
        JacobiAdd(work.round1, work.sum, work.round0); PipeBarrier<PIPE_V>();
        JacobiSubtract(work.round1, accumulator.high, work.round1);
        JacobiSubtract(work.round0, work.round0, work.product); PipeBarrier<PIPE_V>();
    } else {
        JacobiAdd(work.sum, accumulator.high, work.product); PipeBarrier<PIPE_V>();
        JacobiSubtract(work.round0, work.sum, accumulator.high); PipeBarrier<PIPE_V>();
        JacobiSubtract(work.round1, work.sum, work.round0); PipeBarrier<PIPE_V>();
        JacobiSubtract(work.round1, accumulator.high, work.round1);
        JacobiSubtract(work.round0, work.product, work.round0); PipeBarrier<PIPE_V>();
    }
    JacobiAdd(work.round0, work.round1, work.round0); PipeBarrier<PIPE_V>();
    if constexpr (subtract) {
        JacobiSubtract(accumulator.low, accumulator.low, work.error);
    } else {
        JacobiAdd(accumulator.low, accumulator.low, work.error);
    }
    PipeBarrier<PIPE_V>();
    JacobiAdd(accumulator.low, accumulator.low, work.round0); PipeBarrier<PIPE_V>();
    // Rotate disjoint UB views instead of copying an entire vector.
    const auto previous = accumulator.high;
    accumulator.high = work.sum;
    work.sum = previous;
}

__aicore__ inline void JacobiResidualTerm(LocalTensor<float> storage, uint32_t coefficient,
    JacobiComplexView value, JacobiResidualPair &residual, JacobiProductWork &work)
{
    auto a = JacobiParts(storage,
        {JacobiArray(storage, coefficient), JacobiArray(storage, coefficient + 1U)}, kJacobiCoefficientRealHigh);
    auto b = JacobiParts(storage, value, kJacobiValueRealHigh);
    JacobiSplitPair(a, b);
    JacobiAccumulateProduct<true>(residual.real, a.real, b.real, work);
    JacobiAccumulateProduct<false>(residual.real, a.imag, b.imag, work);
    JacobiAccumulateProduct<true>(residual.imag, a.real, b.imag, work);
    JacobiAccumulateProduct<true>(residual.imag, a.imag, b.real, work);
}

__aicore__ inline void JacobiSaveRawTile(LocalTensor<float> storage)
{
    for (uint32_t index = 0; index < 4U; ++index) {
        Adds(JacobiArray(storage, kJacobiRawLowerReal + index), JacobiArray(storage, index), 0.0f, kJacobiRows);
    }
    Adds(JacobiArray(storage, kJacobiRawDiagonalReal), JacobiArray(storage, 10), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, kJacobiRawDiagonalImag), JacobiArray(storage, 11), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, kJacobiRawRhsReal), JacobiArray(storage, 4), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, kJacobiRawRhsImag), JacobiArray(storage, 5), 0.0f, kJacobiRows);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiIterationCoefficients(LocalTensor<float> storage)
{
    Adds(JacobiArray(storage, kJacobiNegativeLowerImag), JacobiArray(storage, 1), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, kJacobiNegativeUpperImag), JacobiArray(storage, 3), 0.0f, kJacobiRows);
    PipeBarrier<PIPE_V>();
    for (uint32_t index = 0; index < 4U; ++index) {
        Muls(JacobiArray(storage, index), JacobiArray(storage, index), -1.0f, kJacobiRows);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiCorrectionRhs(LocalTensor<float> storage, uint32_t current)
{
    Adds(JacobiArray(storage, kJacobiInitialReal), JacobiArray(storage, current), 0.0f, kCgtsv2JacobiStride);
    Adds(JacobiArray(storage, kJacobiInitialImag), JacobiArray(storage, current + 1U), 0.0f, kCgtsv2JacobiStride);
    PipeBarrier<PIPE_V>();
    auto indices = JacobiArray(storage, 14).ReinterpretCast<uint32_t>();
    Gather(JacobiArray(storage, 10), JacobiArray(storage, kJacobiInitialReal), indices,
        4U * (kJacobiGuard - 1U), kJacobiRows);
    Gather(JacobiArray(storage, 11), JacobiArray(storage, kJacobiInitialImag), indices,
        4U * (kJacobiGuard - 1U), kJacobiRows);
    Gather(JacobiArray(storage, 12), JacobiArray(storage, kJacobiInitialReal), indices,
        4U * (kJacobiGuard + 1U), kJacobiRows);
    Gather(JacobiArray(storage, 13), JacobiArray(storage, kJacobiInitialImag), indices,
        4U * (kJacobiGuard + 1U), kJacobiRows);
    JacobiResidualPair residual{{JacobiArray(storage, kJacobiResidualRealHigh), JacobiArray(storage, kJacobiResidualRealLow)},
        {JacobiArray(storage, kJacobiResidualImagHigh), JacobiArray(storage, kJacobiResidualImagLow)}};
    JacobiProductWork work{JacobiArray(storage, kJacobiProduct), JacobiArray(storage, kJacobiProductError),
        JacobiArray(storage, kJacobiSum), JacobiArray(storage, kJacobiRound0), JacobiArray(storage, kJacobiRound1)};
    Adds(residual.real.high, JacobiArray(storage, kJacobiRawRhsReal), 0.0f, kJacobiRows);
    Adds(residual.imag.high, JacobiArray(storage, kJacobiRawRhsImag), 0.0f, kJacobiRows);
    Duplicate(residual.real.low, 0.0f, kJacobiRows);
    Duplicate(residual.imag.low, 0.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, kJacobiRows);
    JacobiResidualTerm(storage, kJacobiRawDiagonalReal,
        {JacobiArray(storage, kJacobiInitialReal)[kJacobiGuard], JacobiArray(storage, kJacobiInitialImag)[kJacobiGuard]}, residual, work);
    JacobiResidualTerm(storage, kJacobiRawLowerReal, {JacobiArray(storage, 10), JacobiArray(storage, 11)}, residual, work);
    JacobiResidualTerm(storage, kJacobiRawUpperReal, {JacobiArray(storage, 12), JacobiArray(storage, 13)}, residual, work);
    JacobiAdd(JacobiArray(storage, 4), residual.real.high, residual.real.low);
    JacobiAdd(JacobiArray(storage, 5), residual.imag.high, residual.imag.low); PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    Adds(JacobiArray(storage, 10), JacobiArray(storage, kJacobiRawDiagonalReal), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, 11), JacobiArray(storage, kJacobiRawDiagonalImag), 0.0f, kJacobiRows);
    Adds(JacobiArray(storage, 12), JacobiArray(storage, kJacobiInverseCache), 0.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    JacobiNormalize(JacobiArray(storage, 4), JacobiArray(storage, 5), storage);
    for (uint32_t index = 6; index < 10U; ++index) {
        Duplicate(JacobiArray(storage, index), 0.0f, kCgtsv2JacobiStride);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void JacobiRefinedIteration(LocalTensor<float> storage, uint32_t current, uint32_t next)
{
    auto lr = JacobiArray(storage, 0), li = JacobiArray(storage, 1);
    auto ur = JacobiArray(storage, 2), ui = JacobiArray(storage, 3);
    auto nr = JacobiArray(storage, next)[kJacobiGuard], ni = JacobiArray(storage, next + 1U)[kJacobiGuard];
    auto t0 = JacobiArray(storage, 10), t1 = JacobiArray(storage, 11);
    auto t2 = JacobiArray(storage, 12), t3 = JacobiArray(storage, 13);
    auto indices = JacobiArray(storage, 14).ReinterpretCast<uint32_t>();
    Gather(t0, JacobiArray(storage, current), indices, 4U * (kJacobiGuard - 1U), kJacobiRows);
    Gather(t1, JacobiArray(storage, current + 1U), indices, 4U * (kJacobiGuard - 1U), kJacobiRows);
    Gather(t2, JacobiArray(storage, current), indices, 4U * (kJacobiGuard + 1U), kJacobiRows);
    Gather(t3, JacobiArray(storage, current + 1U), indices, 4U * (kJacobiGuard + 1U), kJacobiRows);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, kJacobiRows);
    JacobiMultiply(nr, lr, t0);
    JacobiMultiply(ni, li, t0); PipeBarrier<PIPE_V>();
    JacobiMultiplyAdd(nr, JacobiArray(storage, kJacobiNegativeLowerImag), t1);
    JacobiMultiplyAdd(ni, lr, t1); PipeBarrier<PIPE_V>();
    JacobiMultiplyAdd(nr, ur, t2);
    JacobiMultiplyAdd(ni, ui, t2); PipeBarrier<PIPE_V>();
    JacobiMultiplyAdd(nr, JacobiArray(storage, kJacobiNegativeUpperImag), t3);
    JacobiMultiplyAdd(ni, ur, t3); PipeBarrier<PIPE_V>();
    JacobiAdd(nr, nr, JacobiArray(storage, 4));
    JacobiAdd(ni, ni, JacobiArray(storage, 5)); PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline uint32_t JacobiIterateHalf(LocalTensor<float> storage)
{
    uint32_t current = 6U, next = 8U;
    for (uint32_t iteration = 0; iteration < kCgtsv2JacobiIters / 2U; ++iteration) {
        JacobiRefinedIteration(storage, current, next);
        const uint32_t previous = current;
        current = next;
        next = previous;
    }
    return current;
}

__aicore__ inline uint32_t JacobiRefineTile(LocalTensor<float> storage)
{
    const uint32_t initial = JacobiIterateHalf(storage);
    JacobiCorrectionRhs(storage, initial);
    const uint32_t corrected = JacobiIterateHalf(storage);
    Add(JacobiArray(storage, corrected)[kJacobiGuard], JacobiArray(storage, corrected)[kJacobiGuard],
        JacobiArray(storage, kJacobiInitialReal)[kJacobiGuard], kJacobiRows);
    Add(JacobiArray(storage, corrected + 1U)[kJacobiGuard], JacobiArray(storage, corrected + 1U)[kJacobiGuard],
        JacobiArray(storage, kJacobiInitialImag)[kJacobiGuard], kJacobiRows); PipeBarrier<PIPE_V>();
    return corrected;
}

__aicore__ inline void JacobiPrepareTile(Buffers &buffers, int64_t offset, int32_t center,
    uint32_t centerCount, LocalTensor<float> storage, JacobiEvents &events)
{
    const int32_t first = center > static_cast<int32_t>(kCgtsv2JacobiIters) ?
        center - static_cast<int32_t>(kCgtsv2JacobiIters) : 0;
    const uint32_t pad = kCgtsv2JacobiIters - static_cast<uint32_t>(center - first);
    const uint32_t available = static_cast<uint32_t>(buffers.m - first);
    const uint32_t count = available < kJacobiRows - pad ? available : kJacobiRows - pad;
    for (uint32_t index = 0; index < 6U; ++index) {
        Duplicate(JacobiArray(storage, index), 0.0f, kCgtsv2JacobiStride);
    }
    Duplicate(JacobiArray(storage, 10), 1.0f, kJacobiRows);
    Duplicate(JacobiArray(storage, 11), 0.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    JacobiLoadCoefficients(buffers, first, count, pad, storage, events);
    JacobiLoadOriginalRhs(buffers, offset, center, first, count, centerCount, pad, storage, events);
    JacobiSaveRawTile(storage);
    auto dr = JacobiArray(storage, 10), di = JacobiArray(storage, 11);
    auto inverse = JacobiArray(storage, 12), temporary = JacobiArray(storage, 13);
    Mul(inverse, dr, dr, kJacobiRows);
    Mul(temporary, di, di, kJacobiRows); PipeBarrier<PIPE_V>();
    Add(inverse, inverse, temporary, kJacobiRows); PipeBarrier<PIPE_V>();
    JacobiAccurateInverse(storage);
    Adds(JacobiArray(storage, kJacobiInverseCache), inverse, 0.0f, kJacobiRows); PipeBarrier<PIPE_V>();
    JacobiNormalize(JacobiArray(storage, 0), JacobiArray(storage, 1), storage);
    JacobiNormalize(JacobiArray(storage, 2), JacobiArray(storage, 3), storage);
    JacobiNormalize(JacobiArray(storage, 4), JacobiArray(storage, 5), storage);
    for (uint32_t index = 6; index < 10U; ++index) {
        Duplicate(JacobiArray(storage, index), 0.0f, kCgtsv2JacobiStride);
    }
    JacobiIterationCoefficients(storage);
    JacobiIndices(storage, 2U, kJacobiRows);
}

__aicore__ inline void JacobiStoreTile(Buffers &buffers, int64_t offset, int32_t center,
    uint32_t count, uint32_t current, LocalTensor<float> storage, JacobiEvents &events)
{
    auto indices = JacobiArray(storage, 10).ReinterpretCast<uint32_t>();
    auto rows = JacobiArray(storage, 12).ReinterpretCast<uint32_t>();
    const uint32_t components = 2U * count;
    CreateVecIndex(indices.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), components);
    PipeBarrier<PIPE_V>();
    ShiftRight(rows, indices, 1U, components); PipeBarrier<PIPE_V>();
    ShiftLeft(rows, rows, 2U, components);
    ShiftLeft(indices, indices, 31U, components); PipeBarrier<PIPE_V>();
    ShiftRight(indices, indices, 31U, components); PipeBarrier<PIPE_V>();
    Muls(indices.ReinterpretCast<int32_t>(), indices.ReinterpretCast<int32_t>(),
        static_cast<int32_t>(4U * kCgtsv2JacobiStride), components); PipeBarrier<PIPE_V>();
    Add(indices.ReinterpretCast<int32_t>(), indices.ReinterpretCast<int32_t>(),
        rows.ReinterpretCast<int32_t>(), components); PipeBarrier<PIPE_V>();
    auto raw = JacobiArray(storage, 15);
    Gather(raw, JacobiArray(storage, current)[kJacobiGuard + kCgtsv2JacobiIters],
        indices, 0U, components);
    SetFlag<HardEvent::V_MTE3>(events.vToStore);
    WaitFlag<HardEvent::V_MTE3>(events.vToStore);
    DataCopyPad(buffers.rhs[2LL * (offset + center)], raw, OneCopy(8U * count));
    SetFlag<HardEvent::MTE3_V>(events.storeToV);
    WaitFlag<HardEvent::MTE3_V>(events.storeToV);
}

__aicore__ inline void SolveJacobiTile(Buffers &buffers, int64_t offset, int32_t center,
    uint32_t count, LocalTensor<float> storage, JacobiEvents &events)
{
    // Forty initial steps, one original-equation residual and forty correction
    // steps have dependency radius at most 79. The existing 80-row original-B
    // halo therefore isolates each center tile without a GM barrier.
    JacobiPrepareTile(buffers, offset, center, count, storage, events);
    const uint32_t current = JacobiRefineTile(storage);
    JacobiStoreTile(buffers, offset, center, count, current, storage, events);
}

__aicore__ inline void SolveJacobi(Buffers &buffers, uint32_t block, uint32_t blocks,
    LocalTensor<float> storage, TPipe &pipe)
{
    JacobiEvents events(pipe);
    if (buffers.parallelSingleRhs) {
        for (int64_t tile = block; tile * kCgtsv2JacobiCenterRows < buffers.m; tile += blocks) {
            const int32_t center = static_cast<int32_t>(tile * kCgtsv2JacobiCenterRows);
            const uint32_t count = static_cast<uint32_t>(buffers.m - center) < kCgtsv2JacobiCenterRows ?
                static_cast<uint32_t>(buffers.m - center) : kCgtsv2JacobiCenterRows;
            SolveJacobiTile(buffers, 0, center, count, storage, events);
        }
        return;
    }
    for (int64_t column = block; column < buffers.n; column += blocks) {
        const int64_t offset = column * buffers.ldb;
        for (int32_t center = 0; center < buffers.m;) {
            const uint32_t count = static_cast<uint32_t>(buffers.m - center) < kCgtsv2JacobiCenterRows ?
                static_cast<uint32_t>(buffers.m - center) : kCgtsv2JacobiCenterRows;
            SolveJacobiTile(buffers, offset, center, count, storage, events);
            center += static_cast<int32_t>(count);
        }
    }
}

template <typename FloatTensor, typename PivotTensor>
__aicore__ inline void FactorTile(Buffers &buffers, int32_t first, int32_t steps,
    FloatTensor &l, FloatTensor &d, FloatTensor &u, FloatTensor &v, PivotTensor &pivots,
    Complex &a, Complex &firstUpper)
{
    for (int32_t row = 0; row < steps; ++row) {
        const int32_t globalRow = first + row;
        const Complex lower = Cgtsv2::Read(l, row);
        const Complex nextDiagonal = Cgtsv2::Read(d, row);
        const Complex nextUpper = globalRow < buffers.m - 2 ?
            Cgtsv2::Read(u, row) : Complex{0.0f, 0.0f};
        const FactorResult result = ComputeFactorRow(buffers, globalRow, a, firstUpper,
            lower, nextDiagonal, nextUpper);
        // Each original lower/D/U entry is consumed before reusing its UB
        // slot for the current output row. Future input slots remain intact.
        WriteFactors(l, d, u, v, pivots, row, globalRow < buffers.m - 2, result);
        a = result.nextDiagonal;
        firstUpper = result.nextUpper;
    }
}

__aicore__ inline void LoadFactorTile(Buffers &buffers, int32_t first, uint32_t steps,
    LocalTensor<float> l, LocalTensor<float> d, LocalTensor<float> u)
{
    DataCopyPad(l, buffers.lower[2LL * (first + 1)], OneCopy(8U * steps), NoPad<float>());
    DataCopyPad(d, buffers.diagonal[2LL * (first + 1)], OneCopy(8U * steps), NoPad<float>());
    DataCopyPad(u, buffers.upper[2LL * (first + 1)], OneCopy(8U * steps), NoPad<float>());
}

__aicore__ inline void StoreFactorTile(Buffers &buffers, int32_t first, uint32_t steps, bool last,
    LocalTensor<float> l, LocalTensor<float> d, LocalTensor<float> u, LocalTensor<float> v,
    LocalTensor<uint32_t> pivots)
{
    const uint32_t finalRows = steps + (last ? 1U : 0U);
    DataCopyPad(buffers.l[2LL * first], l, OneCopy(8U * steps));
    DataCopyPad(buffers.dWork[2LL * first], d, OneCopy(8U * finalRows));
    DataCopyPad(buffers.u[2LL * first], u, OneCopy(8U * steps));
    const int32_t vRemaining = buffers.m - 2 - first;
    const uint32_t vRows = vRemaining < static_cast<int32_t>(steps) ?
        static_cast<uint32_t>(vRemaining) : steps;
    if (vRows != 0) { DataCopyPad(buffers.v[2LL * first], v, OneCopy(8U * vRows)); }
    DataCopyPad(buffers.pivot[first], pivots, OneCopy(4U * finalRows));
}

__aicore__ inline bool CertifyDirectUb(Buffers &buffers, LocalTensor<float> storage,
    uint32_t stride, TPipe &pipe)
{
    auto lower = storage;
    auto diagonal = storage[2U * stride];
    auto upper = storage[4U * stride];
    const auto loaded = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    const auto consumed = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE2));
    for (int32_t first = 0; first < buffers.m;) {
        const uint32_t count = buffers.m - first < static_cast<int32_t>(kCgtsv2FactorTileSteps) ?
            static_cast<uint32_t>(buffers.m - first) : kCgtsv2FactorTileSteps;
        DataCopyPad(lower, buffers.lower[2LL * first], OneCopy(8U * count), NoPad<float>());
        DataCopyPad(diagonal, buffers.diagonal[2LL * first], OneCopy(8U * count), NoPad<float>());
        DataCopyPad(upper, buffers.upper[2LL * first], OneCopy(8U * count), NoPad<float>());
        SetFlag<HardEvent::MTE2_S>(loaded);
        WaitFlag<HardEvent::MTE2_S>(loaded);
        bool eligible = true;
        for (uint32_t row = 0; row < count; ++row) {
            if (!Cgtsv2::DirectRowEligible(Cgtsv2::Read(lower, row),
                Cgtsv2::Read(diagonal, row), Cgtsv2::Read(upper, row))) {
                eligible = false;
                break;
            }
        }
        SetFlag<HardEvent::S_MTE2>(consumed);
        WaitFlag<HardEvent::S_MTE2>(consumed);
        if (!eligible) { return false; }
        first += static_cast<int32_t>(count);
    }
    return true;
}

__aicore__ inline bool CertifyDirectRhsUb(Buffers &buffers, LocalTensor<float> storage,
    uint32_t stride, TPipe &pipe)
{
    // Six stride-sized float regions plus eight result elements fit inside
    // the factor allocation of nine strides (stride is at least eight).
    auto raw = storage;
    auto absolute = storage[2U * stride];
    auto result = storage[4U * stride];
    auto scratch = storage[4U * stride + 8U];
    JacobiEvents events(pipe);
    for (int64_t column = 0; column < buffers.n; ++column) {
        for (int32_t first = 0; first < buffers.m;) {
            const uint32_t count = static_cast<uint32_t>(buffers.m - first) < stride ?
                static_cast<uint32_t>(buffers.m - first) : stride;
            events.BeforeLoad();
            DataCopyPad(raw, buffers.rhs[2LL * (column * buffers.ldb + first)],
                OneCopy(8U * count), NoPad<float>());
            events.Loaded();
            Abs(absolute, raw, 2U * count);
            PipeBarrier<PIPE_V>();
            ReduceMax(result, absolute, scratch, 2U * count, false);
            SetFlag<HardEvent::V_S>(events.vToS);
            WaitFlag<HardEvent::V_S>(events.vToS);
            const float maximum = result.GetValue(0);
            SetFlag<HardEvent::S_V>(events.sToV);
            WaitFlag<HardEvent::S_V>(events.sToV);
            if (!DirectRhsValueEligible(Complex{maximum, 0.0f})) { return false; }
            first += static_cast<int32_t>(count);
        }
    }
    return true;
}

__aicore__ inline void FactorJacobiOrExpanded(Buffers &buffers, LocalTensor<float> storage, TPipe &pipe)
{
    if (TryCertifyJacobi(buffers, storage, pipe)) {
        buffers.status.SetValue(0, Cgtsv2::kJacobi);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(buffers.status);
    } else {
        FactorExpanded(buffers);
    }
}

__aicore__ inline void FactorUb(Buffers &buffers, LocalTensor<float> storage,
    uint32_t stride, TPipe &pipe)
{
    InitializeStatus(buffers);
    // Large UB inputs use the stricter Jacobi certificate. Boundary matrices
    // rejected by it (and nonfinite RHS) go directly to expanded-range solving;
    // they do not fall through to the less restrictive small direct predicate.
    if (buffers.m >= kCgtsv2JacobiMinRows) {
        FactorJacobiOrExpanded(buffers, storage, pipe);
        return;
    }
    buffers.directEligible = CertifyDirectUb(buffers, storage, stride, pipe) &&
        CertifyDirectRhsUb(buffers, storage, stride, pipe);
    if (!buffers.directEligible) { FactorExpanded(buffers); return; }
    auto l = storage;
    auto d = storage[2U * stride];
    auto u = storage[4U * stride];
    auto v = storage[6U * stride];
    auto pivots = storage.ReinterpretCast<uint32_t>()[8U * stride];
    const auto loaded = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    const auto produced = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    const auto stored = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    const auto completed = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    Complex a = Cgtsv2::Read(buffers.diagonal, 0);
    Complex firstUpper = Cgtsv2::Read(buffers.upper, 0);
    for (int32_t first = 0; first < buffers.m - 1;) {
        const int32_t remaining = buffers.m - 1 - first;
        const uint32_t steps = remaining < static_cast<int32_t>(kCgtsv2FactorTileSteps) ?
            static_cast<uint32_t>(remaining) : kCgtsv2FactorTileSteps;
        LoadFactorTile(buffers, first, steps, l, d, u);
        SetFlag<HardEvent::MTE2_S>(loaded);
        WaitFlag<HardEvent::MTE2_S>(loaded);
        FactorTile(buffers, first, static_cast<int32_t>(steps), l, d, u, v, pivots, a, firstUpper);
        const bool last = first + static_cast<int32_t>(steps) == buffers.m - 1;
        if (last) {
            Cgtsv2::Write(d, steps, a);
            pivots.SetValue(steps, static_cast<uint32_t>(buffers.m - 1));
            FinalStatus(buffers, a);
        }
        SetFlag<HardEvent::S_MTE3>(produced);
        WaitFlag<HardEvent::S_MTE3>(produced);
        StoreFactorTile(buffers, first, steps, last, l, d, u, v, pivots);
        SetFlag<HardEvent::MTE3_MTE2>(stored);
        WaitFlag<HardEvent::MTE3_MTE2>(stored);
        first += static_cast<int32_t>(steps);
    }
    SetFlag<HardEvent::MTE3_S>(completed);
    WaitFlag<HardEvent::MTE3_S>(completed);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(buffers.status);
}

__aicore__ inline void RunFactor(Buffers &buffers, const Cgtsv2TilingData &tiling)
{
    if (tiling.useUb == 0) { Factor(buffers); return; }
    // Direct staging is used only below the Jacobi threshold.
    const uint32_t rows = tiling.m >= kCgtsv2JacobiMinRows ? 0U : static_cast<uint32_t>(tiling.m - 1);
    const uint32_t stride = (rows + kCgtsv2UbRowAlignment) &
        ~(kCgtsv2UbRowAlignment - 1U); // One extra final D/pivot element.
    TPipe pipe;
    TBuf<TPosition::VECCALC> local;
    const uint32_t factorBytes = kCgtsv2FactorUbBytesPerRow * stride;
    const uint32_t bytes = tiling.m >= kCgtsv2JacobiMinRows ? kCgtsv2JacobiStorageBytes : factorBytes;
    pipe.InitBuffer(local, bytes);
    FactorUb(buffers, local.Get<float>(), stride, pipe);
}

class Solver {
public:
    __aicore__ inline void Init(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
        GM_ADDR workspace, const Cgtsv2TilingData &tiling)
    {
        buffers_.Init(dl, d, du, B, workspace, tiling);
        block_ = GetBlockIdx();
        blocks_ = tiling.blockNum;
        useUb_ = tiling.useUb != 0;
        if (useUb_) {
            // Five complex arrays plus uint32 pivots: 44*alignedRows bytes.
            // The Host derives this guard from actual UB capacity and headroom.
            const uint32_t rows = tiling.m >= kCgtsv2JacobiMinRows ? 0U : static_cast<uint32_t>(tiling.m);
            stride_ = (rows + kCgtsv2UbRowAlignment - 1U) & ~(kCgtsv2UbRowAlignment - 1U);
            const uint32_t bytes = tiling.m >= kCgtsv2JacobiMinRows ?
                kCgtsv2JacobiStorageBytes : kCgtsv2SolveUbBytesPerRow * stride_;
            pipe_.InitBuffer(local_, bytes);
        }
    }

    __aicore__ inline void Process()
    {
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(buffers_.status);
        const uint32_t flags = buffers_.status.GetValue(0);
        if ((flags & Cgtsv2::kJacobi) != 0) {
            SolveJacobi(buffers_, block_, blocks_, local_.Get<float>(), pipe_);
            return;
        }
        if ((flags & Cgtsv2::kExpanded) != 0) {
            if (block_ == 0) { SolveExpanded(buffers_, flags); }
            return;
        }
        if ((flags & Cgtsv2::kDirectEligible) == 0) {
            SolveRefined(flags);
            return;
        }
        if (!useUb_) { SolveGm(); }
        else { SolveUb(); }
    }

    __aicore__ inline void FactorAndSolve()
    {
        if (useUb_) { FactorUb(buffers_, local_.Get<float>(), stride_, pipe_); }
        else { Factor(buffers_); }
        Process();
    }

private:
    __aicore__ inline bool CopyOriginal(int64_t offset)
    {
        bool finite = true;
        for (int32_t row = 0; row < buffers_.m; ++row) {
            const Complex value = Cgtsv2::Read(buffers_.rhs, offset + row);
            Cgtsv2::Write(buffers_.scratch, row, value);
            finite = finite && Cgtsv2::Finite(value);
        }
        return finite;
    }

    __aicore__ inline bool BuildResidual(int64_t offset)
    {
        for (int32_t row = 0; row < buffers_.m; ++row) {
            const Complex current = Cgtsv2::Read(buffers_.rhs, offset + row);
            Complex previous{0.0f, 0.0f};
            Complex next{0.0f, 0.0f};
            Complex lower{0.0f, 0.0f};
            Complex upper{0.0f, 0.0f};
            if (row != 0) {
                previous = Cgtsv2::Read(buffers_.rhs, offset + row - 1);
                lower = Cgtsv2::Read(buffers_.lower, row);
            }
            if (row + 1 < buffers_.m) {
                next = Cgtsv2::Read(buffers_.rhs, offset + row + 1);
                upper = Cgtsv2::Read(buffers_.upper, row);
            }
            if (!Cgtsv2::Finite(current) || !Cgtsv2::Finite(previous) || !Cgtsv2::Finite(next)) {
                return false;
            }
            const Complex residual = Cgtsv2::Residual(Cgtsv2::Read(buffers_.scratch, row), lower,
                Cgtsv2::Read(buffers_.diagonal, row), upper, previous, current, next);
            Cgtsv2::Write(buffers_.scratch, row, residual);
        }
        return true;
    }

    __aicore__ inline void ApplyCorrection(int64_t offset)
    {
        for (int32_t row = 0; row < buffers_.m; ++row) {
            if (!Cgtsv2::Finite(Cgtsv2::Read(buffers_.scratch, row))) { return; }
        }
        for (int32_t row = 0; row < buffers_.m; ++row) {
            const Complex value = Cgtsv2::Read(buffers_.rhs, offset + row);
            const Complex delta = Cgtsv2::Read(buffers_.scratch, row);
            Cgtsv2::Write(buffers_.rhs, offset + row,
                Complex{value.real + delta.real, value.imag + delta.imag});
        }
    }

    __aicore__ inline void SolveRefined(uint32_t flags)
    {
        // The one m-complex scratch is shared, so this numerical fallback has
        // exactly one writer even when the launch also contains inactive cores.
        if (block_ != 0) { return; }
        for (int64_t column = 0; column < buffers_.n; ++column) {
            const int64_t offset = column * buffers_.ldb;
            const bool finiteRhs = CopyOriginal(offset);
            SolveColumn(buffers_.m, buffers_.l, buffers_.dWork, buffers_.u, buffers_.v,
                buffers_.pivot, buffers_.rhs, offset);
            if (!finiteRhs || (flags & (Cgtsv2::kSingular | Cgtsv2::kNonfinite)) != 0 ||
                !BuildResidual(offset)) {
                continue;
            }
            SolveColumn(buffers_.m, buffers_.l, buffers_.dWork, buffers_.u, buffers_.v,
                buffers_.pivot, buffers_.scratch, 0);
            ApplyCorrection(offset);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(buffers_.rhs);
    }

    __aicore__ inline void SolveGm()
    {
        // GM scalar writes cannot have multiple DCache writers to the same
        // physical cacheline. Host dispatches this fallback with blockNum=1.
        if (block_ != 0) { return; }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(buffers_.dWork);
        for (int64_t column = 0; column < buffers_.n; ++column) {
            SolveColumn(buffers_.m, buffers_.l, buffers_.dWork, buffers_.u, buffers_.v,
                buffers_.pivot, buffers_.rhs, column * buffers_.ldb, true);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(buffers_.rhs);
    }

    __aicore__ inline void SolveUb()
    {
        auto storage = local_.Get<float>();
        auto l = storage;
        auto d = storage[2U * stride_];
        auto u = storage[4U * stride_];
        auto v = storage[6U * stride_];
        auto rhs = storage[8U * stride_];
        auto pivots = local_.Get<uint32_t>()[10U * stride_];
        const uint32_t m = static_cast<uint32_t>(buffers_.m);
        const auto mteToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        const auto sToMte = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        const auto stored = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
        DataCopyPad(l, buffers_.l, OneCopy(8U * (m - 1U)), NoPad<float>());
        DataCopyPad(d, buffers_.dWork, OneCopy(8U * m), NoPad<float>());
        DataCopyPad(u, buffers_.u, OneCopy(8U * (m - 1U)), NoPad<float>());
        DataCopyPad(v, buffers_.v, OneCopy(8U * (m - 2U)), NoPad<float>());
        DataCopyPad(pivots, buffers_.pivot, OneCopy(4U * m), NoPad<uint32_t>());
        for (int64_t column = block_; column < buffers_.n; column += blocks_) {
            const int64_t offset = 2LL * column * buffers_.ldb;
            DataCopyPad(rhs, buffers_.rhs[offset], OneCopy(8U * m), NoPad<float>());
            SetFlag<HardEvent::MTE2_S>(mteToS);
            WaitFlag<HardEvent::MTE2_S>(mteToS);
            SolveColumn(buffers_.m, l, d, u, v, pivots, rhs, 0, true);
            SetFlag<HardEvent::S_MTE3>(sToMte);
            WaitFlag<HardEvent::S_MTE3>(sToMte);
            // The exact valid byte count preserves B[ m : ldb, column ].
            DataCopyPad(buffers_.rhs[offset], rhs, OneCopy(8U * m));
            SetFlag<HardEvent::MTE3_MTE2>(stored);
            WaitFlag<HardEvent::MTE3_MTE2>(stored);
        }
    }

    Buffers buffers_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> local_;
    uint32_t block_ = 0;
    int32_t blocks_ = 1;
    uint32_t stride_ = 0;
    bool useUb_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void cgtsv2_factor_kernel(GM_ADDR dl, GM_ADDR d,
    GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Cgtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_NOT_AIC {
        Buffers buffers;
        buffers.Init(dl, d, du, B, workspace, tiling);
        RunFactor(buffers, tiling);
        ResetMask();
    }
}

extern "C" __global__ __aicore__ void cgtsv2_solve_kernel(GM_ADDR dl, GM_ADDR d,
    GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Cgtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_NOT_AIC {
        Solver solver;
        solver.Init(dl, d, du, B, workspace, tiling);
        solver.Process();
        ResetMask();
    }
}

extern "C" __global__ __aicore__ void cgtsv2_single_rhs_kernel(GM_ADDR dl, GM_ADDR d,
    GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Cgtsv2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_NOT_AIC {
        Solver solver;
        solver.Init(dl, d, du, B, workspace, tiling);
        solver.FactorAndSolve();
        ResetMask();
    }
}

void cgtsv2_factor_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace, const Cgtsv2TilingData &tiling, void *stream)
{
    cgtsv2_factor_kernel<<<1, nullptr, stream>>>(dl, d, du, B, workspace, tiling);
}

void cgtsv2_solve_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace, const Cgtsv2TilingData &tiling, void *stream)
{
    cgtsv2_solve_kernel<<<static_cast<uint32_t>(tiling.blockNum), nullptr, stream>>>(
        dl, d, du, B, workspace, tiling);
}

void cgtsv2_single_rhs_kernel_do(GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B,
    GM_ADDR workspace, const Cgtsv2TilingData &tiling, void *stream)
{
    cgtsv2_single_rhs_kernel<<<1, nullptr, stream>>>(dl, d, du, B, workspace, tiling);
}
