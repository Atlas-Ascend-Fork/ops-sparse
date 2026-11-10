/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_SOLVE_H_
#define SPSV_ARCH22_SPSV_SOLVE_H_
#include "spsv_scaled_solution.h"
#include "spsv_compact_solve.h"

namespace spsv_a2 {
using namespace AscendC;

template <int32_t Components>
class Solve {
public:
    __aicore__ inline Solve(TPipe& pipe, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, SpsvA2Tiling tiling)
        : t_(tiling), input_(x), output_(y), workspace_(workspace)
    {
        row_.Init(pipe, workspace == nullptr ? nullptr : workspace + t_.rowOffset, t_.m + 1);
        col_.Init(pipe, workspace == nullptr ? nullptr : workspace + t_.colOffset, t_.nnz);
        value_.Init(pipe, workspace == nullptr ? nullptr : workspace + t_.valueOffset, t_.nnz * Components);
        x_.Init(pipe, x, t_.m * Components);
        y_.Init(pipe, y, t_.m * Components);
        alpha_.Init(pipe, reinterpret_cast<GM_ADDR>(t_.alphaDevice), Components);
        if (t_.unit && t_.precisionOffset != 0)
            math_.Init(pipe);
    }
    __aicore__ inline void Run()
    {
        input_ = PreserveAliasedInput(x_, input_, output_, workspace_, t_);
        if (t_.unit && t_.precisionOffset != 0) {
            WideSolution<Components, spsv_a2_ub::kResidentCapacity> planned(t_, row_, col_, value_, x_, y_);
            if (planned.UnitMain(ReadAlpha(), input_, output_, workspace_, math_))
                return;
            y_.Rebind(output_, t_.m * Components);
        }
        if (t_.m * Components <= spsv_a2_ub::kResidentCapacity) {
            if (t_.unit) RunResident<true>();
            else RunResident<false>();
            return;
        }
        Complex alpha = ReadAlpha();
        bool forward = t_.transpose ? !t_.lower : t_.lower;
        for (int64_t step = 0; step < t_.m; ++step) {
            int64_t row = forward ? step : t_.m - 1 - step;
            int64_t offset = row * Components;
            Complex source{x_.Read(offset), Components == 2 ? x_.Read(offset + 1) : 0};
            bool scaled = NeedsScaling(source) || NeedsScaling(alpha);
            bool diagonalReplay = false;
            Complex rhs = Components == 1 ? Complex{source.real * alpha.real, 0} : Multiply(alpha, source);
            Complex sum{0, 0};
            Complex correction{0, 0};
            Complex magnitudes{0, 0};
            TrackProductMagnitude<Components, true>(alpha, source, magnitudes);
            spsv_dd::Complex diagonal{{0, 0}, {0, 0}};
            if (t_.nnz != 0)
                AccumulateRow(row, sum, correction, diagonal, magnitudes, scaled, diagonalReplay);
            rhs.real = SubtractCompensated(rhs.real, sum.real, correction.real);
            rhs.imag = SubtractCompensated(rhs.imag, sum.imag, correction.imag);
            bool productReplay = RequiresProductReplay(row, rhs, magnitudes);
            Complex result = t_.unit ? rhs : Divide(rhs, diagonal, Components);
            if (RequiresWideReplay(row, productReplay, magnitudes, diagonal, t_.unit, result)) {
                ReplayWide(alpha);
                return;
            }
            scaled |= productReplay || diagonalReplay;
            scaled |= NeedsScaling(rhs);
            scaled |= NeedsScaling(result);
            if (scaled && !ResolveRow(step, row, alpha, source, result))
                return;
            y_.Write(offset, result.real);
            if (Components == 2)
                y_.Write(offset + 1, result.imag);
        }
        y_.Flush();
    }

private:
    __aicore__ inline Complex ReadAlpha()
    {
        Complex alpha{t_.alphaReal, t_.alphaImag};
        if (t_.alphaDevice != 0) {
            alpha.real = alpha_.Read(0);
            alpha.imag = Components == 2 ? alpha_.Read(1) : 0;
        }
        return alpha;
    }
    __aicore__ inline bool RequiresWideReplay(int64_t row, bool productReplay, Complex magnitudes,
                                              spsv_dd::Complex diagonal, bool unit, Complex result)
    {
        // Range protection does not bound rounding in previously stored rows.
        if (!productReplay)
            return false;
        if (t_.nnz > kMaxBoundedTerms &&
            static_cast<int64_t>(row_.Read(row + 1)) - row_.Read(row) + 1 > kMaxBoundedTerms)
            return true;
        return NeedsDependencyReplay(magnitudes, diagonal, unit, result);
    }
    __aicore__ inline void ReplayWide(Complex alpha)
    {
        WideSolution<Components, spsv_a2_ub::kResidentCapacity> solution(t_, row_, col_, value_, x_, y_);
        solution.Run(alpha, input_, output_, workspace_);
    }
    __aicore__ inline bool RequiresProductReplay(int64_t row, Complex rhs, Complex magnitudes)
    {
        int64_t terms = t_.nnz + 1;
        if (terms > kMaxBoundedTerms)
            terms = static_cast<int64_t>(row_.Read(row + 1)) - row_.Read(row) + 1;
        return NeedsExactProducts(rhs, magnitudes, terms);
    }
    __aicore__ inline bool ResolveRow(int64_t step, int64_t row, Complex alpha, Complex source, Complex& result)
    {
        auto exact = ExactRow<Components>(row, alpha, source, t_.unit, row_, col_, value_, y_);
        if (NeedsScaledDependency(exact)) {
            ContinueScaled(step, alpha);
            return false;
        }
        result = RoundPair(spsv_scaled::ToPair(exact));
        return true;
    }
    __aicore__ inline void ContinueScaled(int64_t step, Complex alpha)
    {
        ScaledSolution<Components, spsv_a2_ub::kResidentCapacity> solution(t_, row_, col_, value_, x_, y_);
        solution.Continue(step, alpha, input_, output_, workspace_);
    }
    __aicore__ inline void AccumulateRow(int64_t row, Complex& sum, Complex& correction, spsv_dd::Complex& diagonal,
                                         Complex& magnitudes, bool& scaled, bool& diagonalReplay)
    {
        int32_t begin = row_.Read(row);
        int32_t end = row_.Read(row + 1);
        bool diagonalSeen = false;
        for (int64_t p = begin; p < end; ++p) {
            int32_t col = col_.Read(p);
            Complex a{value_.Read(p * Components), Components == 2 ? value_.Read(p * 2 + 1) : 0};
            scaled |= NeedsScaling(a);
            if (col == row) {
                if (!t_.unit)
                    AccumulateDiagonal(a, diagonal, Components, diagonalSeen, diagonalReplay);
                continue;
            }
            Complex previous{y_.Read(static_cast<int64_t>(col) * Components),
                             Components == 2 ? y_.Read(static_cast<int64_t>(col) * 2 + 1) : 0};
            scaled |= NeedsScaling(previous);
            TrackProductMagnitude<Components, true>(a, previous, magnitudes);
            Complex term = Components == 1 ? Complex{a.real * previous.real, 0} : Multiply(a, previous);
            Accumulate(term.real, sum.real, correction.real, scaled);
            if (Components == 2)
                Accumulate(term.imag, sum.imag, correction.imag, scaled);
        }
        scaled = scaled || NeedsScaling({diagonal.real.hi, diagonal.imag.hi}) ||
            !Finite(correction.real) || !Finite(correction.imag);
    }
    template <bool Unit, bool CheckPrevious>
    __aicore__ inline void AccumulateResidentRow(int32_t row, LocalTensor<float> y, Complex& sum, Complex& correction,
                                                 spsv_dd::Complex& diagonal, Complex& magnitudes, bool& scaled,
                                                 bool& diagonalReplay)
    {
        int32_t end = row_.Read(row + 1);
        bool diagonalSeen = false;
        for (int32_t p = row_.Read(row); p < end;) {
            uint32_t colAvailable, valueAvailable;
            LocalTensor<int32_t> cols = col_.ReadSpan(p, colAvailable);
            LocalTensor<float> values = value_.ReadSpan(static_cast<int64_t>(p) * Components, valueAvailable);
            uint32_t count = static_cast<uint32_t>(end - p);
            if (count > colAvailable) count = colAvailable;
            if (count > valueAvailable / Components) count = valueAvailable / Components;
            for (uint32_t i = 0; i < count; ++i) {
                int32_t col = cols.GetValue(i);
                Complex a{values.GetValue(i * Components),
                    Components == 2 ? values.GetValue(i * 2 + 1) : 0};
                scaled |= (Unit && !CheckPrevious) ? NeedsCheckedValue(a) : NeedsScaling(a);
                if (col == row) {
                    if constexpr (!Unit)
                        AccumulateDiagonal(a, diagonal, Components, diagonalSeen, diagonalReplay);
                    continue;
                }
                uint32_t previousOffset = static_cast<uint32_t>(col) * Components;
                Complex previous{y.GetValue(previousOffset),
                    Components == 2 ? y.GetValue(previousOffset + 1) : 0};
                if constexpr (CheckPrevious) scaled |= NeedsScaling(previous);
                TrackProductMagnitude<Components, true>(a, previous, magnitudes);
                Complex term = Components == 1 ? Complex{a.real * previous.real, 0} : Multiply(a, previous);
                Accumulate<!Unit || CheckPrevious>(term.real, sum.real, correction.real, scaled);
                if (Components == 2) {
                    Accumulate<!Unit || CheckPrevious>(term.imag, sum.imag, correction.imag, scaled);
                }
            }
            p += static_cast<int32_t>(count);
        }
        scaled = scaled || NeedsScaling({diagonal.real.hi, diagonal.imag.hi}) ||
            !Finite(correction.real) || !Finite(correction.imag);
    }
    template <bool Unit>
    __aicore__ inline void RunResident()
    {
        // Same serial triangular solve and Neumaier order as Run. Hoist
        // per-entry window checks when all dependent Y values stay in UB.
        LocalTensor<float> y = y_.WritableWhole();
        Complex alpha = ReadAlpha();
        bool forward = t_.transpose ? !t_.lower : t_.lower;
        // Regular components are bounded by 2^50, so each complex product is
        // bounded by 2^101. At most 2^20 terms keep the sum and compensation
        // finite. Extreme coefficients replay exactly; extreme dependencies
        // and larger matrices retain the checked accumulation path.
        constexpr int64_t maxUncheckedEntries = 1LL << 20;
        bool checkDependencies = !Unit || t_.nnz > maxUncheckedEntries;
        for (int64_t step = 0; step < t_.m; ++step) {
            int32_t row = static_cast<int32_t>(forward ? step : t_.m - 1 - step);
            uint32_t offset = static_cast<uint32_t>(row) * Components;
            Complex source{x_.Read(offset), Components == 2 ? x_.Read(offset + 1) : 0};
            // Exact diagonal reduction does not establish dependency accuracy.
            bool scaled = NeedsScaling(source) || NeedsScaling(alpha);
            bool diagonalReplay = false;
            Complex rhs = Components == 1 ? Complex{source.real * alpha.real, 0} : Multiply(alpha, source);
            Complex sum{0, 0};
            Complex correction{0, 0};
            Complex magnitudes{0, 0};
            TrackProductMagnitude<Components, true>(alpha, source, magnitudes);
            spsv_dd::Complex diagonal{{0, 0}, {0, 0}};
            if (t_.nnz != 0) {
                if (checkDependencies) {
                    AccumulateResidentRow<Unit, true>(row, y, sum, correction, diagonal, magnitudes, scaled,
                                                      diagonalReplay);
                } else {
                    AccumulateResidentRow<Unit, false>(row, y, sum, correction, diagonal, magnitudes, scaled,
                                                       diagonalReplay);
                }
            }
            rhs.real = SubtractCompensated(rhs.real, sum.real, correction.real);
            rhs.imag = SubtractCompensated(rhs.imag, sum.imag, correction.imag);
            bool productReplay = RequiresProductReplay(row, rhs, magnitudes);
            Complex result = Unit ? rhs : Divide(rhs, diagonal, Components);
            if (RequiresWideReplay(row, productReplay, magnitudes, diagonal, Unit, result)) {
                ReplayWide(alpha);
                return;
            }
            scaled |= productReplay || diagonalReplay;
            scaled |= NeedsScaling(rhs);
            scaled |= NeedsScaling(result);
            if (scaled && !ResolveRow(step, row, alpha, source, result))
                return;
            if constexpr (Unit)
                checkDependencies = checkDependencies || NeedsCheckedValue(result);
            y.SetValue(offset, result.real);
            if (Components == 2)
                y.SetValue(offset + 1, result.imag);
        }
        y_.Flush();
    }
    spsv_unit_vector::Math math_;
    SpsvA2Tiling t_;
    GM_ADDR input_;
    GM_ADDR output_;
    GM_ADDR workspace_;
    UbCache<int32_t, spsv_a2_ub::kRowCapacity> row_;
    UbCache<int32_t, spsv_a2_ub::kColumnCapacity> col_;
    UbCache<float, spsv_a2_ub::kValueCapacity> value_;
    UbCache<float, spsv_a2_ub::kInputCapacity> x_;
    UbCache<float, spsv_a2_ub::kResidentCapacity> y_;
    UbCache<float, spsv_a2_ub::kAlphaCapacity> alpha_;
    static_assert(decltype(row_)::kBytes + decltype(col_)::kBytes + decltype(value_)::kBytes + decltype(x_)::kBytes +
                          decltype(y_)::kBytes + decltype(alpha_)::kBytes + spsv_unit_vector::kBytes ==
                      spsv_a2_ub::kRequiredBytes,
                  "Solve allocations must match the shared UB budget");
};

// Analysis proves the selected structure has no off-diagonal entries. These
// rows are independent, including duplicate diagonals, singular rows and UNIT.
template <int32_t Components>
class DiagonalSolve {
public:
    __aicore__ inline DiagonalSolve(TPipe &pipe, GM_ADDR x, GM_ADDR y,
        GM_ADDR workspace, SpsvA2Tiling tiling, int64_t begin, int64_t rows)
        : t_(tiling), begin_(begin), rows_(rows)
    {
        row_.Init(pipe, workspace == nullptr ? nullptr : workspace + t_.rowOffset, t_.m + 1);
        value_.Init(pipe, workspace == nullptr ? nullptr : workspace + t_.valueOffset, t_.nnz * Components);
        x_.Init(pipe, x + begin * Components * sizeof(float), rows * Components);
        y_.Init(pipe, y + begin * Components * sizeof(float), rows * Components);
        alpha_.Init(pipe, reinterpret_cast<GM_ADDR>(t_.alphaDevice), Components);
    }
    __aicore__ inline void Run()
    {
        Complex alpha{t_.alphaReal, t_.alphaImag};
        if (t_.alphaDevice != 0) {
            alpha.real = alpha_.Read(0);
            alpha.imag = Components == 2 ? alpha_.Read(1) : 0;
        }
        for (int64_t localRow = 0; localRow < rows_; ++localRow) {
            int64_t offset = localRow * Components;
            Complex source{x_.Read(offset), Components == 2 ? x_.Read(offset + 1) : 0};
            Complex rhs = source;
            rhs = Components == 1 ? Complex{rhs.real * alpha.real, 0} : Multiply(alpha, rhs);
            spsv_dd::Complex diagonal{{0, 0}, {0, 0}};
            bool scaled = NeedsScaling(source) || NeedsScaling(alpha);
            if (!t_.unit && t_.nnz != 0) {
                int64_t row = begin_ + localRow;
                bool diagonalSeen = false;
                for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
                    Complex a{value_.Read(p * Components),
                        Components == 2 ? value_.Read(p * Components + 1) : 0};
                    scaled = scaled || NeedsScaling(a);
                    AccumulateDiagonal(a, diagonal, Components, diagonalSeen, scaled);
                }
            }
            scaled = scaled || NeedsScaling({diagonal.real.hi, diagonal.imag.hi});
            scaled |= NeedsScaling(rhs);
            Complex result = t_.unit ? rhs : Divide(rhs, diagonal, Components);
            if (scaled) result = ScaledProductQuotient(alpha, source,
                ScaledDiagonal(begin_ + localRow), t_.unit, Components);
            y_.Write(offset, result.real);
            if (Components == 2) y_.Write(offset + 1, result.imag);
        }
        y_.Flush();
    }
private:
    __aicore__ inline spsv_scaled::Complex ScaledDiagonal(int64_t row)
    {
        spsv_scaled::ExactDiagonalSum<Components> diagonal;
        if (!t_.unit && t_.nnz != 0) {
            for (int64_t p = row_.Read(row); p < row_.Read(row + 1); ++p) {
                diagonal.Add(value_.Read(p * Components),
                    Components == 2 ? value_.Read(p * Components + 1) : 0);
            }
        }
        return diagonal.Value();
    }
    SpsvA2Tiling t_;
    int64_t begin_, rows_;
    UbCache<int32_t, 256> row_;
    UbCache<float, 512> value_;
    UbCache<float, 256> x_, y_;
    UbCache<float, 8> alpha_;
    static_assert(decltype(row_)::kBytes + decltype(value_)::kBytes + decltype(x_)::kBytes +
        decltype(y_)::kBytes + decltype(alpha_)::kBytes <= spsv_a2_ub::kRequiredBytes,
        "Diagonal solve exceeds the UB budget");
};

template <int32_t Components>
__aicore__ inline void RunSolve(TPipe &pipe, GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, SpsvA2Tiling tiling)
{
    if (tiling.diagonalMatrix) {
        // 128-row-aligned owners prevent adjacent blocks from sharing output
        // DMA/cache lines. Each owner's X/Y cache is bounded to its own range.
        int64_t tiles = (tiling.m + 127) / 128;
        int64_t chunk = (tiles + tiling.blocks - 1) / tiling.blocks * 128;
        int64_t begin = GetBlockIdx() * chunk;
        if (begin >= tiling.m) return;
        int64_t rows = tiling.m - begin;
        if (rows > chunk) rows = chunk;
        DiagonalSolve<Components> kernel(pipe, x, y, workspace, tiling, begin, rows);
        kernel.Run();
    } else if (tiling.m * Components <= spsv_a2_ub::kCompactCapacity) {
        // Both compact paths retain the complete high/low solution in UB.
        // A nonconverged or nonfinite iteration falls back to direct solve.
        if constexpr (Components == 2) {
            if (tiling.unit && tiling.m > UnitBlockJacobiSolve::kBlockRows) {
                UnitBlockJacobiSolve iterative(pipe, x, y, workspace, tiling);
                if (iterative.TryRun()) return;
            }
        }
        AccurateSolve<Components> kernel(pipe, x, y, workspace, tiling);
        kernel.Run();
    } else {
        Solve<Components> kernel(pipe, x, y, workspace, tiling);
        kernel.Run();
    }
}

} // namespace spsv_a2

#endif // SPSV_ARCH22_SPSV_SOLVE_H_
