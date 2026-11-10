/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_COMPACT_SOLVE_H_
#define SPSV_COMPACT_SOLVE_H_
#include "spsv_scaled_solution.h"

namespace spsv_a2 {
using namespace AscendC;

template <int32_t Components>
class AccurateSolve {
public:
    __aicore__ inline AccurateSolve(TPipe& pipe, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, SpsvA2Tiling tiling)
        : t_(tiling), input_(x), output_(y), workspace_(workspace)
    {
        row_.Init(pipe, workspace + t_.rowOffset, t_.m + 1);
        col_.Init(pipe, workspace + t_.colOffset, t_.nnz);
        value_.Init(pipe, workspace + t_.valueOffset, t_.nnz * Components);
        x_.Init(pipe, x, t_.m * Components);
        y_.Init(pipe, y, t_.m * Components);
        alpha_.Init(pipe, reinterpret_cast<GM_ADDR>(t_.alphaDevice), Components);
        pipe.InitBuffer(lowBuffer_, spsv_a2_ub::kLowBytes);
        low_ = lowBuffer_.Get<float>();
    }
    __aicore__ inline void Run()
    {
        input_ = PreserveAliasedInput(x_, input_, output_, workspace_, t_);
        spsv_dd::Complex alpha{{t_.alphaReal, 0}, {t_.alphaImag, 0}};
        if (t_.alphaDevice != 0) {
            alpha.real.hi = alpha_.Read(0);
            alpha.imag.hi = Components == 2 ? alpha_.Read(1) : 0;
        }
        bool forward = t_.transpose ? !t_.lower : t_.lower;
        for (int64_t step = 0; step < t_.m; ++step) {
            int64_t row = forward ? step : t_.m - 1 - step;
            int64_t offset = row * Components;
            Complex source{x_.Read(offset), Components == 2 ? x_.Read(offset + 1) : 0};
            spsv_dd::Complex rhs = ToPair(source);
            bool scaled = NeedsScaling(source) || NeedsScaling({alpha.real.hi, alpha.imag.hi});
            if (Components == 1) rhs.real = spsv_dd::Mul(alpha.real, rhs.real);
            else rhs = spsv_dd::Mul(alpha, rhs);
            spsv_dd::Complex diagonal{{0, 0}, {0, 0}};
            if (AccumulateRow(row, rhs, diagonal, scaled)) {
                WideSolution<Components, spsv_a2_ub::kCompactCapacity> solution(t_, row_, col_, value_, x_, y_);
                solution.Run(Complex{alpha.real.hi, alpha.imag.hi}, input_, output_, workspace_);
                return;
            }
            if (!t_.unit) {
                if (Components == 1) rhs.real = spsv_dd::Div(rhs.real, diagonal.real);
                else rhs = spsv_dd::Div(rhs, diagonal);
            }
            if (scaled && !ResolveRow(step, row, {alpha.real.hi, alpha.imag.hi}, source, rhs))
                return;
            Store(offset, rhs.real);
            if (Components == 2) Store(offset + 1, rhs.imag);
        }
        y_.Flush();
    }
private:
    __aicore__ inline bool ResolveRow(int64_t step, int64_t row, Complex alpha, Complex source,
                                      spsv_dd::Complex& result)
    {
        auto exact = ExactRow<Components>(row, alpha, source, t_.unit, row_, col_, value_, y_, &low_);
        if (NeedsScaledDependency(exact)) {
            ScaledSolution<Components, spsv_a2_ub::kCompactCapacity> solution(t_, row_, col_, value_, x_, y_);
            solution.Continue(step, alpha, input_, output_, workspace_, &low_);
            return false;
        }
        result = spsv_scaled::ToPair(exact);
        return true;
    }
    __aicore__ inline bool AccumulateRow(int64_t row, spsv_dd::Complex& rhs, spsv_dd::Complex& diagonal, bool& scaled)
    {
        bool diagonalSeen = false;
        Complex largest{0, 0};
        for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
            int32_t col = col_.Read(p);
            spsv_dd::Complex a{{value_.Read(p * Components), 0},
                {Components == 2 ? value_.Read(p * 2 + 1) : 0, 0}};
            scaled = scaled || NeedsScaling({a.real.hi, a.imag.hi});
            if (col == row) {
                if (!t_.unit) AccumulateDiagonal({a.real.hi, a.imag.hi}, diagonal,
                    Components, diagonalSeen, scaled);
                continue;
            }
            int64_t previous = static_cast<int64_t>(col) * Components;
            spsv_dd::Complex y{{y_.Read(previous), low_.GetValue(previous)},
                {Components == 2 ? y_.Read(previous + 1) : 0,
                    Components == 2 ? low_.GetValue(previous + 1) : 0}};
            scaled = scaled || NeedsScaling({y.real.hi, y.imag.hi});
            TrackProductMagnitude<Components>({a.real.hi, a.imag.hi}, {y.real.hi, y.imag.hi}, largest);
            if (Components == 1) rhs.real = spsv_dd::Sub(rhs.real, spsv_dd::Mul(a.real, y.real));
            else rhs = spsv_dd::Sub(rhs, spsv_dd::Mul(a, y));
        }
        bool exactResidual = NeedsExactResidual({rhs.real.hi, rhs.imag.hi}, largest);
        bool wide = exactResidual && NeedsCompactWidePrecision(largest, diagonal, t_.unit);
        scaled = scaled || NeedsScaling({diagonal.real.hi, diagonal.imag.hi}) ||
                 NeedsScaling({rhs.real.hi, rhs.imag.hi}) || !Finite(rhs.real.hi) || !Finite(rhs.imag.hi) ||
                 exactResidual;
        return wide;
    }
    __aicore__ inline void Store(int64_t offset, spsv_dd::Real value)
    {
        spsv_dd::Real normalized = (spsv_scaled::Bits(value.hi) & 0x7fffffffu) < 0x00800000u ?
            spsv_dd::Real{RoundReal(value), 0} : spsv_dd::Sum(value.hi, value.lo);
        y_.Write(offset, normalized.hi);
        low_.SetValue(offset, normalized.lo);
    }
    SpsvA2Tiling t_;
    GM_ADDR input_;
    GM_ADDR output_;
    GM_ADDR workspace_;
    UbCache<int32_t, spsv_a2_ub::kRowCapacity> row_;
    UbCache<int32_t, spsv_a2_ub::kColumnCapacity> col_;
    UbCache<float, spsv_a2_ub::kValueCapacity> value_;
    UbCache<float, spsv_a2_ub::kCompactCapacity> y_;
    UbCache<float, spsv_a2_ub::kInputCapacity> x_;
    UbCache<float, spsv_a2_ub::kAlphaCapacity> alpha_;
    TBuf<TPosition::VECCALC> lowBuffer_;
    LocalTensor<float> low_;
    static_assert(decltype(row_)::kBytes + decltype(col_)::kBytes + decltype(value_)::kBytes +
        decltype(x_)::kBytes + decltype(y_)::kBytes + decltype(alpha_)::kBytes + spsv_a2_ub::kLowBytes ==
        spsv_a2_ub::kAccurateBytes, "Accurate solve allocations must match the shared UB budget");
};

// Compact unit-triangular solve using the actual normalized input. Stopping
// and high/low verification depend only on A, X, alpha and FP32 precision.
class UnitBlockJacobiSolve {
public:
    static constexpr int32_t kBlockRows = 4;

    __aicore__ inline UnitBlockJacobiSolve(TPipe &pipe, GM_ADDR x, GM_ADDR y,
        GM_ADDR workspace, SpsvA2Tiling tiling) : t_(tiling)
    {
        row_.Init(pipe, workspace + t_.rowOffset, t_.m + 1);
        col_.Init(pipe, workspace + t_.colOffset, t_.nnz);
        value_.Init(pipe, workspace + t_.valueOffset, t_.nnz * 2);
        x_.Init(pipe, x, t_.m * 2);
        y_.Init(pipe, y, t_.m * 2);
        alpha_.Init(pipe, reinterpret_cast<GM_ADDR>(t_.alphaDevice), 2);
        pipe.InitBuffer(iterBuffer_, spsv_a2_ub::kIterateBytes);
        iter_ = iterBuffer_.Get<float>();
    }

    __aicore__ inline bool TryRun()
    {
        if (!InitializeIterates()) return false;
        const int64_t block = kBlockRows;
        int32_t previous = 0;
        for (int64_t iteration = 1; iteration <= 8 * t_.m; ++iteration) {
            const int32_t next = 1 - previous;
            bool converged = true;
            for (int64_t begin = 0; begin < t_.m; begin += block) {
                int64_t end = begin + block;
                if (end > t_.m) end = t_.m;
                if (!SolveBlock(begin, end, previous, next)) return false;
                RelaxBlock(begin, end, previous, next, converged);
            }
            if (converged) return VerifyAndCommit(next, previous);
            previous = next;
        }
        return false; // Caller retains the existing direct high/low solver.
    }

private:
    static constexpr int32_t kCapacity = spsv_a2_ub::kCompactCapacity;
    // 32 * epsilon(float32). Absolute floor avoids a relative zero test.
    static constexpr float kRelativeTolerance = 0x1p-18f;

    __aicore__ inline bool InitializeIterates()
    {
        alphaValue_ = {{t_.alphaReal, 0}, {t_.alphaImag, 0}};
        if (t_.alphaDevice != 0) {
            alphaValue_.real.hi = alpha_.Read(0);
            alphaValue_.imag.hi = alpha_.Read(1);
        }
        if (!Finite(alphaValue_.real.hi) || !Finite(alphaValue_.imag.hi) ||
            NeedsScaling({alphaValue_.real.hi, alphaValue_.imag.hi}))
            return false;
        for (int64_t p = 0; p < t_.m * 2; ++p) {
            float source = x_.Read(p);
            if (!Finite(source) || spsv_scaled::NeedsScaling(source))
                return false;
            for (int32_t bank = 0; bank < 2; ++bank) Set(bank, p, {0, 0});
        }
        return true;
    }
    __aicore__ inline bool SolveBlock(int64_t begin, int64_t end, int32_t previous, int32_t next)
    {
        const bool forward = t_.transpose ? !t_.lower : t_.lower;
        for (int64_t step = 0; step < end - begin; ++step) {
            const int64_t row = forward ? begin + step : end - 1 - step;
            // Avoid repeatedly iterating rows longer than the compact cache.
            // The direct high/low path retains their individual product errors.
            if (row_.Read(row + 1) - row_.Read(row) > kCapacity)
                return false;
            auto rhs = Rhs(row);
            Complex largest{0, 0};
            for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
                const int32_t col = col_.Read(p);
                if (col == row) continue;
                auto a = Coefficient(p);
                if (!Finite(a.real.hi) || !Finite(a.imag.hi) || NeedsScaling({a.real.hi, a.imag.hi}))
                    return false;
                const int32_t bank = col >= begin && col < end ? next : previous;
                auto y = GetComplex(bank, col);
                if (NeedsScaling({y.real.hi, y.imag.hi}))
                    return false;
                TrackProductMagnitude<2>({a.real.hi, a.imag.hi}, {y.real.hi, y.imag.hi}, largest);
                rhs = spsv_dd::Sub(rhs, spsv_dd::Mul(a, y));
            }
            if (!Finite(rhs.real.hi) || !Finite(rhs.imag.hi) || NeedsExactResidual({rhs.real.hi, rhs.imag.hi}, largest))
                return false;
            SetComplex(next, row, rhs);
        }
        return true;
    }
    __aicore__ inline void RelaxBlock(int64_t begin, int64_t end, int32_t previous,
        int32_t next, bool &converged)
    {
        // Keep each block's unrelaxed result until its internal dependencies are consumed.
        for (int64_t row = begin; row < end; ++row) {
            for (int32_t part = 0; part < 2; ++part) {
                const int64_t offset = row * 2 + part;
                const auto old = Get(previous, offset);
                const auto delta = spsv_dd::Mul(spsv_dd::Sub(Get(next, offset), old), {0.5f, 0});
                const auto relaxed = spsv_dd::Add(old, delta);
                Set(next, offset, relaxed);
                if (!WithinPrecision(delta, relaxed.hi)) converged = false;
            }
        }
    }

    __aicore__ inline bool WithinPrecision(spsv_dd::Real delta, float output)
    {
        if (!Finite(delta.hi) || !Finite(delta.lo) || !Finite(output)) return false;
        float scale = Abs(output);
        if (scale < 1.0f) scale = 1.0f;
        return Abs(delta.hi + delta.lo) <= kRelativeTolerance * scale;
    }

    __aicore__ inline spsv_dd::Real Get(int32_t bank, int64_t offset)
    {
        const int64_t base = bank * (2 * kCapacity) + offset;
        return {iter_.GetValue(base), iter_.GetValue(base + kCapacity)};
    }
    __aicore__ inline void Set(int32_t bank, int64_t offset, spsv_dd::Real value)
    {
        value = spsv_dd::Sum(value.hi, value.lo);
        const int64_t base = bank * (2 * kCapacity) + offset;
        iter_.SetValue(base, value.hi);
        iter_.SetValue(base + kCapacity, value.lo);
    }
    __aicore__ inline spsv_dd::Complex GetComplex(int32_t bank, int64_t row)
    {
        return {Get(bank, row * 2), Get(bank, row * 2 + 1)};
    }
    __aicore__ inline void SetComplex(int32_t bank, int64_t row, spsv_dd::Complex value)
    {
        Set(bank, row * 2, value.real);
        Set(bank, row * 2 + 1, value.imag);
    }
    __aicore__ inline spsv_dd::Complex Rhs(int64_t row)
    {
        return spsv_dd::Mul(alphaValue_, spsv_dd::Complex{{x_.Read(row * 2), 0}, {x_.Read(row * 2 + 1), 0}});
    }
    __aicore__ inline spsv_dd::Complex Coefficient(int64_t p)
    {
        return {{value_.Read(p * 2), 0}, {value_.Read(p * 2 + 1), 0}};
    }

    __aicore__ inline bool VerifyAndCommit(int32_t candidate, int32_t refined)
    {
        // Verify the rounded public output, not just the unrounded iterate.
        for (int64_t p = 0; p < t_.m * 2; ++p) Set(candidate, p, {Get(candidate, p).hi, 0});
        const bool forward = t_.transpose ? !t_.lower : t_.lower;
        bool verified = true;
        for (int64_t step = 0; step < t_.m; ++step) {
            const int64_t row = forward ? step : t_.m - 1 - step;
            auto rhs = Rhs(row);
            Complex largest{0, 0};
            for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
                const int32_t col = col_.Read(p);
                if (col == row) continue;
                auto a = Coefficient(p);
                auto y = GetComplex(refined, col);
                TrackProductMagnitude<2>({a.real.hi, a.imag.hi}, {y.real.hi, y.imag.hi}, largest);
                rhs = spsv_dd::Sub(rhs, spsv_dd::Mul(a, y));
            }
            if (!Finite(rhs.real.hi) || !Finite(rhs.imag.hi) || NeedsExactResidual({rhs.real.hi, rhs.imag.hi}, largest))
                return false;
            SetComplex(refined, row, rhs);
            for (int32_t part = 0; part < 2; ++part) {
                const int64_t offset = row * 2 + part;
                // A^-1*(b-A*y) = refined-y. A single internal high/low
                // verification pass avoids reliance on residual size alone.
                const auto correction = spsv_dd::Sub(Get(refined, offset), Get(candidate, offset));
                if (!WithinPrecision(correction, Get(candidate, offset).hi)) verified = false;
            }
        }
        const int32_t selected = verified ? candidate : refined;
        for (int64_t p = 0; p < t_.m * 2; ++p) y_.Write(p, Get(selected, p).hi);
        y_.Flush();
        return true;
    }

    SpsvA2Tiling t_;
    spsv_dd::Complex alphaValue_;
    UbCache<int32_t, spsv_a2_ub::kRowCapacity> row_;
    UbCache<int32_t, spsv_a2_ub::kColumnCapacity> col_;
    UbCache<float, spsv_a2_ub::kValueCapacity> value_;
    UbCache<float, spsv_a2_ub::kInputCapacity> x_;
    UbCache<float, spsv_a2_ub::kCompactCapacity> y_;
    UbCache<float, spsv_a2_ub::kAlphaCapacity> alpha_;
    TBuf<TPosition::VECCALC> iterBuffer_;
    LocalTensor<float> iter_;
    static_assert(decltype(row_)::kBytes + decltype(col_)::kBytes + decltype(value_)::kBytes +
        decltype(x_)::kBytes + decltype(y_)::kBytes + decltype(alpha_)::kBytes + spsv_a2_ub::kIterateBytes ==
        spsv_a2_ub::kIterativeBytes, "Iterative solve allocations must match the shared UB budget");
};

} // namespace spsv_a2
#endif
