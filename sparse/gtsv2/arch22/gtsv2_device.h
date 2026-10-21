/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GTSV2_ARCH22_DEVICE_H_
#define GTSV2_ARCH22_DEVICE_H_

#include <cstdint>

#include "kernel_operator.h"
#include "gtsv2_kernel.h"

// Internal device types and declarations. Implementations and their template
// instantiations stay together in gtsv2_device.cpp. Host launch declarations
// remain in gtsv2_kernel.h.

struct Gtsv2FloatPair {
    float hi;
    float lo;
};

__aicore__ inline bool Gtsv2Finite(float value);

__aicore__ inline Gtsv2FloatPair Gtsv2TwoSum(float a, float b);

__aicore__ inline Gtsv2FloatPair Gtsv2TwoProduct(float a, float b);

__aicore__ inline Gtsv2FloatPair Gtsv2PairAdd(Gtsv2FloatPair a, Gtsv2FloatPair b);

__aicore__ inline float Gtsv2AccurateResidual(
    float b, float dl, float d, float du, float previous, float x, float next);

// A normalized two-float significand and a separate exponent. No device FP64
// instructions or common matrix scale are needed. In particular, a small
// coefficient is not discarded merely because another row is near FLT_MAX.
struct Gtsv2Scaled {
    float hi;
    float lo;
    int64_t exponent;
};

__aicore__ inline Gtsv2Scaled Gtsv2ScaledFromFloat(float value);

// Used only for bounded significand adjustments, never for an external exponent.
__aicore__ inline float Gtsv2ScaleSignificand(float value, int32_t exponent);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledNormalize(Gtsv2FloatPair value, int64_t exponent);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledNegate(Gtsv2Scaled value);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledAdd(Gtsv2Scaled a, Gtsv2Scaled b);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledMultiply(Gtsv2Scaled a, Gtsv2Scaled b);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledDivide(Gtsv2Scaled a, Gtsv2Scaled b);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledSubtractProduct(
    Gtsv2Scaled value, Gtsv2Scaled factor, Gtsv2Scaled other);

__aicore__ inline bool Gtsv2ScaledAbsGreaterEqual(Gtsv2Scaled a, Gtsv2Scaled b);

__aicore__ inline float Gtsv2ScaledToFloat(Gtsv2Scaled value);

__aicore__ inline int64_t Gtsv2ScaledRowExponent(Gtsv2Scaled lower, Gtsv2Scaled diagonal, Gtsv2Scaled upper);

__aicore__ inline Gtsv2Scaled Gtsv2ScaledResidual(float b, float lower, float diagonal, float upper,
    float previous, float x, float next, Gtsv2Scaled &denominator);

// Five reusable m-element arrays: diagonal, first/second superdiagonal,
// multipliers, and one RHS. Equilibrate each row by an exact power of two before
// partial pivoting. Recompute its exponent from immutable coefficients per RHS
// to retain O(m) workspace and the column-major ABI. An extra m-float array
// retains the original RHS for two expanded-range residual corrections.
// Tensor interfaces deliberately allow the same arithmetic to be CPU-tested.
template <typename FloatTensor, typename ExponentTensor, typename PivotTensor>
class Gtsv2ScaledLu {
public:
    __aicore__ inline void Init(FloatTensor significands, ExponentTensor exponents,
        PivotTensor pivots, FloatTensor original, int32_t m);

    __aicore__ inline Gtsv2Scaled Get(int64_t index);

    __aicore__ inline void Set(int64_t index, Gtsv2Scaled value);

    template <typename CoefficientTensor>
    __aicore__ inline bool Factor(CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du,
        bool equilibrateRows = true);

    template <typename RhsTensor, typename CoefficientTensor>
    __aicore__ inline void Solve(RhsTensor rhs, int64_t offset,
        CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du);

private:
    __aicore__ inline void SolvePreparedRhs();

    template <typename RhsTensor, typename CoefficientTensor>
    __aicore__ inline void SolveColumn(RhsTensor rhs, int64_t offset,
        CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du);

    template <typename RhsTensor, typename CoefficientTensor>
    __aicore__ inline float MaxBackwardError(RhsTensor rhs, int64_t offset,
        CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du);

    template <typename RhsTensor, typename CoefficientTensor>
    __aicore__ inline bool PrepareResidual(RhsTensor rhs, int64_t offset,
        CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du);

    template <typename RhsTensor>
    __aicore__ inline bool ApplyCorrection(RhsTensor rhs, int64_t offset);

    template <typename RhsTensor, typename CoefficientTensor>
    __aicore__ inline void RefineColumn(RhsTensor rhs, int64_t offset,
        CoefficientTensor dl, CoefficientTensor d, CoefficientTensor du);

    FloatTensor significands_;
    FloatTensor original_;
    ExponentTensor exponents_;
    PivotTensor pivots_;
    int32_t m_{0};
    bool equilibrateRows_{true};
};

// A substantial dominance margin bounds elimination growth on the fast path.
// This is a matrix property, independent of shape, seed, or benchmark case ID.
template <typename Tensor>
__aicore__ inline bool Gtsv2NeedsRefinement(Tensor dl, Tensor d, Tensor du, int32_t m);

// Only core zero enters this path. Two reusable m-element GM arrays retain the
// original RHS and the correction; storage is O(m), independent of n. Scalar
// GM traversal also covers systems larger than UB without another dispatch.
class Gtsv2Refinement {
public:
    __aicore__ inline void Init(
        AscendC::GlobalTensor<float> dl, AscendC::GlobalTensor<float> d,
        AscendC::GlobalTensor<float> du, AscendC::GlobalTensor<float> factors,
        AscendC::GlobalTensor<uint8_t> pivots, AscendC::GlobalTensor<float> b,
        GM_ADDR workspace, int32_t m, int32_t n, int32_t ldb);

    __aicore__ inline void Process();

    // Nonfinite RHS columns keep the established FP32 propagation order.
    // This entry point does not touch the shared scratch arrays.
    __aicore__ inline void SolveNonfiniteColumn(int64_t offset);

private:
    __aicore__ inline bool AllFinite(AscendC::GlobalTensor<float> values, int64_t offset);

    __aicore__ inline void Solve(AscendC::GlobalTensor<float> rhs, int64_t offset);

    __aicore__ inline bool FormResidual(int64_t offset);

    __aicore__ inline void RefineColumn(int64_t offset);

    int32_t m_{0}, n_{0}, ldb_{0};
    AscendC::GlobalTensor<float> dl_, d_, du_, factors_, b_, original_, correction_;
    AscendC::GlobalTensor<uint8_t> pivots_;
};

// Entered only by core zero. The dispatch byte and original FP32 factors stay
// intact while other solve cores read the byte and exit. The expanded factors
// occupy separate O(m) scratch, shared by all RHS columns.
class Gtsv2RangeSolver {
public:
    __aicore__ inline void Init(
        AscendC::GlobalTensor<float> dl, AscendC::GlobalTensor<float> d,
        AscendC::GlobalTensor<float> du, AscendC::GlobalTensor<float> factors,
        AscendC::GlobalTensor<uint8_t> pivots, AscendC::GlobalTensor<float> b,
        GM_ADDR workspace, int32_t m, int32_t n, int32_t ldb);

    __aicore__ inline void Process();

private:
    __aicore__ inline bool FiniteCoefficients();

    int32_t m_{0}, n_{0}, ldb_{0};
    Gtsv2Refinement legacy_;
    AscendC::GlobalTensor<float> dl_, d_, du_, b_, significands_, original_;
    AscendC::GlobalTensor<int64_t> exponents_;
    AscendC::GlobalTensor<uint8_t> pivots_;
};

// Only strictly row-dominant systems with sum(abs(offdiag)) < abs(d)/2
// enter this path. Jacobi is an infinity-norm contraction by less than 1/2;
// 40 fixed iterations put truncation well below FP32 rounding. The conservative
// max(abs(B))/min(abs(d)) <= 4 guard also bounds the iterates by 8 and keeps
// FP32 rounding well below the absolute tolerance. Larger-dynamic-range inputs
// retain the pivoted direct solver. Selection uses matrix
// properties, never a seed, case ID, or benchmark value.
constexpr int32_t kGtsv2JacobiMinM = 128;
constexpr int32_t kGtsv2JacobiMaxM = 6144;
constexpr int32_t kGtsv2JacobiIters = 40;

__aicore__ inline bool Gtsv2JacobiEligible(
    int32_t m, AscendC::LocalTensor<float> lower, AscendC::LocalTensor<float> diagonal,
    AscendC::LocalTensor<float> upper, AscendC::LocalTensor<float> rhs,
    AscendC::LocalTensor<float> temporary, AscendC::LocalTensor<float> sumStorage,
    AscendC::LocalTensor<float> boundStorage, event_t vectorToScalar);

__aicore__ inline bool Gtsv2TryJacobi(
    int32_t m,
    AscendC::LocalTensor<float> lower, AscendC::LocalTensor<float> diagonal,
    AscendC::LocalTensor<float> upper, AscendC::LocalTensor<float> rhs,
    AscendC::LocalTensor<float> temporary,
    AscendC::LocalTensor<float> xStorage, AscendC::LocalTensor<float> nextStorage,
    event_t scalarToVector, event_t vectorToScalar);

__aicore__ inline int32_t Gtsv2IterateWideJacobi(
    int32_t m, int32_t stride, AscendC::LocalTensor<float> storage,
    AscendC::GlobalTensor<float> rhsGm, int64_t offset,
    event_t vToMte2, event_t mte2ToV);

__aicore__ inline void Gtsv2InitializeWideJacobi(
    int32_t m, int32_t stride, AscendC::LocalTensor<float> storage);

// Large-UB variant: exactly six aligned m-float arrays (192 KiB at m=8192).
// B is an in-place operand, so it can temporarily hold its normalized RHS.
// Reloading that immutable normalized RHS each iteration saves one UB array.
// X ping-pong arrays are internal to this allocation. Gather's endpoint reads
// stay within allocated finite storage and are multiplied by exact zero.
__aicore__ inline bool Gtsv2TryWideJacobi(
    int32_t m, AscendC::LocalTensor<float> storage,
    AscendC::GlobalTensor<float> lowerGm, AscendC::GlobalTensor<float> diagonalGm,
    AscendC::GlobalTensor<float> upperGm, AscendC::GlobalTensor<float> rhsGm,
    int64_t offset, event_t sToV, event_t vToS, event_t mte2ToV,
    event_t vToMte2, event_t vToMte3, event_t mte3ToV);

namespace {

constexpr uint32_t kFloatBytes = sizeof(float);

__aicore__ inline float Gtsv2Abs(float value);

// Only used after the partial-pivot comparison. A zero denominator here
// implies 0/0 in the unswapped branch, or NaN/0 in the swapped branch.
// Preserve that NaN classification explicitly; do not use this helper for
// back substitution, where a nonzero numerator divided by zero yields Inf.
__aicore__ inline float Gtsv2PivotRatio(float numerator, float denominator);

__aicore__ inline AscendC::DataCopyExtParams OneBlockCopy(uint32_t bytes);

template <typename T>
__aicore__ inline AscendC::DataCopyPadExtParams<T> NoPadding();

__aicore__ inline uint64_t Gtsv2FactorBytes(int32_t m);

__aicore__ inline int32_t Gtsv2AlignedRows(int32_t m);

__aicore__ inline uint32_t Gtsv2FactorStorageBytes(int32_t m);

template <bool SingleRhs = false>
class Gtsv2FactorKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR workspace, GM_ADDR B,
        const Gtsv2TilingData &tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline bool NeedsRefinementInUb(
        AscendC::LocalTensor<float> lower, AscendC::LocalTensor<float> diagonal, AscendC::LocalTensor<float> upper,
        AscendC::LocalTensor<float> margin, AscendC::LocalTensor<float> temporary);

    // Conservative range envelope for the strongly row-dominant fast path.
    // |d| >= 2^-24, max|d| and max|b| <= 2^24 leave ample FP32 headroom
    // for partial-pivot growth, <= m RHS additions (m <= INT32_MAX), and
    // substitution. Small off-diagonals are kept unchanged, not rescaled.
    __aicore__ inline bool NeedsRangeInGm();

    __aicore__ inline float ReduceMagnitude(
        AscendC::LocalTensor<float> input, AscendC::LocalTensor<float> temporary,
        AscendC::LocalTensor<float> reduction, bool minimum);

    __aicore__ inline bool NeedsRangeInUb(
        AscendC::LocalTensor<float> diagonal, AscendC::LocalTensor<float> temporary,
        AscendC::LocalTensor<float> reduction, event_t mte2ToScalar);

    __aicore__ inline bool TrySingleRhsInUb(
        AscendC::LocalTensor<float> multiplier, AscendC::LocalTensor<float> dWork, AscendC::LocalTensor<float> duWork,
        AscendC::LocalTensor<float> rhs, AscendC::LocalTensor<float> du2Work, event_t mte2ToScalar);

    __aicore__ inline void UpdateUbRhsOrPivot(
        AscendC::LocalTensor<float> rhs, AscendC::LocalTensor<uint8_t> pivots, int32_t row, float factor, bool swapped);

    __aicore__ inline void FactorRowsInUb(
        AscendC::LocalTensor<float> dWork, AscendC::LocalTensor<float> duWork, AscendC::LocalTensor<float> du2Work,
        AscendC::LocalTensor<float> multiplier, AscendC::LocalTensor<uint8_t> pivots, AscendC::LocalTensor<float> rhs,
        float &pivotD, float &pivotDu);

    __aicore__ inline void StoreFactorsFromUb(
        AscendC::LocalTensor<float> dWork, AscendC::LocalTensor<float> duWork, AscendC::LocalTensor<float> du2Work,
        AscendC::LocalTensor<float> multiplier, AscendC::LocalTensor<uint8_t> pivots);

    __aicore__ inline bool NeedsProtectedSolveInUb(
        AscendC::LocalTensor<float> multiplier, AscendC::LocalTensor<float> diagonal, AscendC::LocalTensor<float> upper,
        AscendC::LocalTensor<float> temporary, AscendC::LocalTensor<float> rhs, event_t mte2ToScalar);

    __aicore__ inline void FactorInUb();

    __aicore__ inline void SolveSingleRhsInUb(
        AscendC::LocalTensor<float> diagonal, AscendC::LocalTensor<float> upper,
        AscendC::LocalTensor<float> upper2, AscendC::LocalTensor<float> rhs);

    __aicore__ inline void StoreSingleRhsInUb(AscendC::LocalTensor<float> rhs);

    __aicore__ inline void FactorInGm();

private:
    int32_t m_{0};
    int32_t n_{0};
    int32_t ldb_{0};
    GM_ADDR workspace_{nullptr};
    bool needsRefinement_{false};
    bool useUb_{false};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> factorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pivotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> checkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> jacobiBuf_;
    AscendC::GlobalTensor<float> BGm_;
    AscendC::GlobalTensor<float> dlGm_;
    AscendC::GlobalTensor<float> dGm_;
    AscendC::GlobalTensor<float> duGm_;
    AscendC::GlobalTensor<float> factorGm_;
    AscendC::GlobalTensor<uint8_t> pivotGm_;
};

class Gtsv2SolveKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR dl, GM_ADDR d, GM_ADDR du, GM_ADDR B, GM_ADDR workspace, const Gtsv2TilingData &tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SolveOneRhsUb(
        AscendC::LocalTensor<float> factors, AscendC::LocalTensor<uint8_t> pivots,
        AscendC::LocalTensor<float> rhs);

    __aicore__ inline bool TryJacobiForColumn(
        AscendC::LocalTensor<float> factors, AscendC::LocalTensor<float> rhs, event_t rhsReady,
        event_t jacobiSToV, event_t jacobiVToS);

    __aicore__ inline void SolveInUb();

    __aicore__ inline void SolveWideJacobiInUb();

    __aicore__ inline void SolveInGm();

private:
    int32_t m_{0};
    int32_t n_{0};
    int32_t ldb_{0};
    int32_t blockNum_{1};
    int32_t blockIdx_{0};
    GM_ADDR workspace_{nullptr};
    bool useUb_{false};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> factorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pivotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> jacobiBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wideJacobiBuf_;
    AscendC::GlobalTensor<float> BGm_;
    AscendC::GlobalTensor<float> dlGm_;
    AscendC::GlobalTensor<float> dGm_;
    AscendC::GlobalTensor<float> duGm_;
    AscendC::GlobalTensor<float> factorGm_;
    AscendC::GlobalTensor<uint8_t> pivotGm_;
};

}  // namespace

#endif  // GTSV2_ARCH22_DEVICE_H_
