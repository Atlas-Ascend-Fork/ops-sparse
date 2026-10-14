/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_MATH_H
#define SPSM_ARCH22_MATH_H
#include "kernel_operator.h"
#include "../spsm_tiling_data.h"
#include "spsm_dispatch.h"

namespace spsm_a2 {
using namespace AscendC;
struct Scalar { float real{0}; float imag{0}; };
// Preprocessing completed on the same stream before either solve kernel starts.
template<bool Complex>
__aicore__ inline bool PrepareSolve(SpsmA2TilingData &tiling,
    GlobalTensor<int32_t> &status, GlobalTensor<float> &alpha)
{
    if (tiling.rows <= 0 || tiling.rhs <= 0) { return false; }
    for (int32_t block = 0; block < tiling.statusBlocks; ++block) {
        if (status.GetValue(block * kSpsmA2StatusStride) != 0) { return false; }
    }
    if (!DispatchReady(status, tiling.statusBlocks, tiling.packedFallback)) { return false; }
    if (tiling.deviceAlpha) {
        tiling.alpha = alpha.GetValue(0);
        if constexpr (Complex) { tiling.alphaImag = alpha.GetValue(1); }
    }
    return true;
}

}

namespace spsm_a2 {
// Two FP32 components retain the rounding residual across dependent rows.
// Products use a bit split: this target's multiply-add is not an IEEE FMA.
// All RHS arithmetic, splitting and error transforms are vector operations.
template<bool Complex, int32_t Width = kSpsmA2RhsTile> class CompensatedMath {
    static constexpr int32_t kTile = Width;
    static constexpr int32_t kParts = Complex ? 2 : 1;
    enum Slot {
        ACC_H = 0, ACC_L = 2, INPUT_H = 4, INPUT_L = 6,
        PRODUCT_H = 8, PRODUCT_L = 9, X_H = 10, X_L = 11,
        COEF_H = 12, COEF_L = 13, COEF = 14, TMP = 15,
        SUM = 16, VIRTUAL = 17, ERROR = 18, RESIDUAL = 19,
        NORMAL_H = 20, NORMAL_L = 21, ABS_H = 22, EXPONENT_MASK = 23, NUMERATOR = 24, ESTIMATE = 28
    };
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(data_, 32 * kTile * sizeof(float));
        pipe->InitBuffer(splitMask_, kTile * sizeof(int32_t));
        pipe->InitBuffer(finiteMask_, kTile * sizeof(int32_t));
        pipe->InitBuffer(offset_, kTile * sizeof(int32_t));
        pipe->InitBuffer(packOffset_, 2 * kTile * sizeof(int32_t));
        Duplicate(data_.Get<float>(), 0.0f, 32 * kTile);
        Barrier();
        Duplicate(splitMask_.Get<int32_t>(), -4096, kTile);
        Duplicate(V(EXPONENT_MASK).template ReinterpretCast<int32_t>(), 0x7f800000, kTile);
        InitOffsets(pipe);
    }
    __aicore__ inline void Begin(LocalTensor<float> input, Scalar alpha, int32_t count)
    {
        Unpack(input, INPUT_H, count);
        Duplicate(V(INPUT_L), 0.0f, kParts * kTile);
        Barrier();
        MultiplyInput(alpha, count);
    }
    __aicore__ inline void ReadHigh(LocalTensor<float> input, int32_t count)
    {
        Unpack(input, INPUT_H, count);
    }
    __aicore__ inline void Subtract(LocalTensor<float> low, Scalar value, int32_t count)
    {
        Unpack(low, INPUT_L, count);
        SubtractInput(value, count);
    }
    __aicore__ inline void Divide(Scalar diag, int32_t count)
    {
        CopyPair(NUMERATOR, ACC_H);
        CopyPair(INPUT_H, ACC_H);
        Scalar inverse = Reciprocal(diag);
        // A complex quotient can fit FP32 even when one product with the
        // reciprocal overflows before the real/imaginary cancellation. Form
        // the estimate at half scale when the reciprocal amplifies inputs.
        // Rescale both compensated components before residual refinement.
        bool halfEstimate = false;
        if constexpr (Complex) {
            halfEstimate = FiniteScalar(inverse.real) && FiniteScalar(inverse.imag) &&
                (inverse.real > 1.0f || inverse.real < -1.0f ||
                 inverse.imag > 1.0f || inverse.imag < -1.0f);
        }
        Scalar estimateInverse = halfEstimate ?
            Scalar{inverse.real * 0.5f, inverse.imag * 0.5f} : inverse;
        MultiplyInput(estimateInverse, count);
        if (halfEstimate) {
            Muls(V(ACC_H), V(ACC_H), 2.0f, 4 * kTile);
            Barrier();
        }
        // The ordinary IEEE estimate is the result for a nonfinite divisor.
        // Residual refinement would otherwise evaluate infinity times zero.
        if (!FiniteScalar(diag.real) || !FiniteScalar(diag.imag)) { return; }
        CopyPair(ESTIMATE, ACC_H);
        CopyPair(INPUT_H, ESTIMATE);
        CopyPair(ACC_H, NUMERATOR);
        SubtractInput(diag, count);
        CopyPair(INPUT_H, ACC_H);
        MultiplyInput(inverse, count);
        for (int32_t part = 0; part < kParts; ++part) {
            AddPair(part, V(ESTIMATE + part), V(ESTIMATE + 2 + part), count);
        }
        PreserveNonfiniteEstimate(count);
    }
    __aicore__ inline void DivideUnit(int32_t count)
    {
        // Apply the complex unit reciprocal with ordinary complex arithmetic,
        // matching the CPU triangular reference's INF/NAN propagation. The
        // exact identity preserves both components of every finite pair.
        CopyPair(INPUT_H, ACC_H);
        MultiplyInput({1.0f, 0.0f}, count);
    }
    __aicore__ inline void Pack(LocalTensor<float> output, int32_t count)
    {
        PackPart(output, ACC_H, count);
    }
    __aicore__ inline void PackLow(LocalTensor<float> output, int32_t count)
    {
        PackPart(output, ACC_L, count);
    }
    // Experimental long-row dot product: independent terms occupy vector lanes.
    // The tree below is fixed by the logical entry order, never by completion order.
    __aicore__ inline void NegativeProducts(LocalTensor<float> high, LocalTensor<float> low,
        LocalTensor<float> real, LocalTensor<float> imag, int32_t count)
    {
        Unpack(high, INPUT_H, count);
        Unpack(low, INPUT_L, count);
        ProductVector(0, real, true, count);
        DataCopy(V(ACC_H), V(PRODUCT_H), kTile);
        DataCopy(V(ACC_L), V(PRODUCT_L), kTile);
        Barrier();
        if constexpr (Complex) {
            ProductVector(1, imag, false, count);
            AddPair(0, V(PRODUCT_H), V(PRODUCT_L), count);
            ProductVector(1, real, true, count);
            DataCopy(V(ACC_H + 1), V(PRODUCT_H), kTile);
            DataCopy(V(ACC_L + 1), V(PRODUCT_L), kTile);
            Barrier();
            ProductVector(0, imag, true, count);
            AddPair(1, V(PRODUCT_H), V(PRODUCT_L), count);
        }
        ClearNonfiniteLow(count);
    }
    __aicore__ inline void FoldTerms(LocalTensor<uint32_t> even, LocalTensor<uint32_t> odd,
        int32_t terms, int32_t rhs)
    {
        while (terms > 1) {
            terms /= 2;
            int32_t count = terms * rhs;
            for (int32_t part = 0; part < kParts; ++part) {
                Gather(V(NUMERATOR + part), V(ACC_H + part), even, 0, count);
                Gather(V(NUMERATOR + 2 + part), V(ACC_L + part), even, 0, count);
                Gather(V(ESTIMATE + part), V(ACC_H + part), odd, 0, count);
                Gather(V(ESTIMATE + 2 + part), V(ACC_L + part), odd, 0, count);
            }
            Barrier();
            CopyPair(ACC_H, NUMERATOR);
            for (int32_t part = 0; part < kParts; ++part) {
                AddPair(part, V(ESTIMATE + part), V(ESTIMATE + 2 + part), count);
            }
        }
    }
    __aicore__ inline LocalTensor<float> High(int32_t part) { return V(ACC_H + part); }
    __aicore__ inline LocalTensor<float> Low(int32_t part) { return V(ACC_L + part); }
    __aicore__ inline void AddReduced(int32_t part, LocalTensor<float> high,
        LocalTensor<float> low, int32_t count)
    {
        AddPair(part, high, low, count);
    }
    // FP32 diagonal solves can put different rows in adjacent lanes. Keep the
    // same compensated reciprocal refinement, with one divisor per lane.
    __aicore__ inline void DivideRealVector(LocalTensor<float> diagonal,
        LocalTensor<float> inverse, int32_t count)
    {
        CopyPair(NUMERATOR, ACC_H);
        CopyPair(INPUT_H, ACC_H);
        MultiplyRealVector(inverse, count);
        CopyPair(ESTIMATE, ACC_H);
        CopyPair(INPUT_H, ESTIMATE);
        CopyPair(ACC_H, NUMERATOR);
        ProductVector(0, diagonal, true, count);
        AddPair(0, V(PRODUCT_H), V(PRODUCT_L), count);
        CopyPair(INPUT_H, ACC_H);
        MultiplyRealVector(inverse, count);
        AddPair(0, V(ESTIMATE), V(ESTIMATE + 2), count);
        PreserveNonfiniteEstimate(count);
        // Scalar Divide returns the initial IEEE estimate for an INF/NAN divisor.
        FiniteMask(diagonal, count);
        BlendHigh(V(ACC_H), V(ACC_H), V(ESTIMATE), count);
        And(V(ACC_L).template ReinterpretCast<uint16_t>(), V(ACC_L).template ReinterpretCast<uint16_t>(),
            finiteMask_.Get<uint16_t>(), count * 2);
        Barrier();
    }
private:
    __aicore__ inline void MultiplyRealVector(LocalTensor<float> coefficient, int32_t count)
    {
        ProductVector(0, coefficient, false, count);
        DataCopy(V(ACC_H), V(PRODUCT_H), kTile);
        DataCopy(V(ACC_L), V(PRODUCT_L), kTile);
        Barrier();
        ClearNonfiniteLow(count);
    }
    __aicore__ inline LocalTensor<float> V(int32_t slot) { return data_.Get<float>()[slot * kTile]; }
    __aicore__ inline void Barrier() { PipeBarrier<PIPE_V>(); }
    __aicore__ inline void CopyPair(int32_t dst, int32_t src)
    {
        DataCopy(V(dst), V(src), 4 * kTile);
        Barrier();
    }
    __aicore__ inline void InitOffsets(TPipe *pipe)
    {
        auto offsets = offset_.Get<int32_t>();
        CreateVecIndex(offsets, 0, kTile);
        Barrier();
        Muls(offsets, offsets, 8, kTile);
        auto packed = packOffset_.Get<int32_t>();
        for (int32_t lane = 0; lane < 8; ++lane) {
            packed.SetValue(lane, (lane / 2 + (lane % 2) * kTile) * 4);
        }
        auto event = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event);
        for (int32_t group = 1; group < kTile / 4; ++group) {
            Adds(packed[group * 8], packed, group * 16, 8);
        }
        Barrier();
    }
    __aicore__ inline void Unpack(LocalTensor<float> input, int32_t slot, int32_t count)
    {
        if constexpr (Complex) {
            Gather(V(slot), input, offset_.Get<uint32_t>(), 0, count);
            Gather(V(slot + 1), input, offset_.Get<uint32_t>(), 4, count);
        } else {
            DataCopy(V(slot), input, (count + 7) / 8 * 8);
        }
        Barrier();
    }
    __aicore__ inline void PackPart(LocalTensor<float> output, int32_t slot, int32_t count)
    {
        if constexpr (Complex) {
            Gather(output, V(slot), packOffset_.Get<uint32_t>(), 0, 2 * count);
        } else {
            DataCopy(output, V(slot), (count + 7) / 8 * 8);
        }
    }
    __aicore__ inline void FiniteMask(LocalTensor<float> high, int32_t count)
    {
        // Exponent 255 denotes INF/NAN. Form one full-word mask per lane
        // without temporary CMPMASK storage or FP comparison of extreme values.
        And(finiteMask_.Get<uint16_t>(), high.template ReinterpretCast<uint16_t>(),
            V(EXPONENT_MASK).template ReinterpretCast<uint16_t>(), count * 2);
        Barrier();
        Adds(finiteMask_.Get<int32_t>(), finiteMask_.Get<int32_t>(), -0x7f800000, count);
        Barrier();
        ShiftRight(finiteMask_.Get<int32_t>(), finiteMask_.Get<int32_t>(), 31, count);
        Barrier();
    }
    __aicore__ inline void Product(int32_t part, float coefficient, int32_t count)
    {
        auto x = V(INPUT_H + part);
        Duplicate(V(COEF), coefficient, count);
        Barrier();
        And(V(X_H).template ReinterpretCast<uint16_t>(), x.template ReinterpretCast<uint16_t>(),
            splitMask_.Get<uint16_t>(), count * 2);
        And(V(COEF_H).template ReinterpretCast<uint16_t>(), V(COEF).template ReinterpretCast<uint16_t>(),
            splitMask_.Get<uint16_t>(), count * 2);
        Muls(V(PRODUCT_H), x, coefficient, count);
        Barrier();
        Sub(V(X_L), x, V(X_H), count);
        Sub(V(COEF_L), V(COEF), V(COEF_H), count);
        Barrier();
        ProductResidual(part, coefficient, count);
    }
    __aicore__ inline void ProductHighResidual(int32_t count)
    {
        Mul(V(PRODUCT_L), V(X_H), V(COEF_H), count);
        Barrier();
        Sub(V(PRODUCT_L), V(PRODUCT_L), V(PRODUCT_H), count);
        Mul(V(TMP), V(X_H), V(COEF_L), count);
        Barrier();
        Add(V(PRODUCT_L), V(PRODUCT_L), V(TMP), count);
        Mul(V(TMP), V(X_L), V(COEF_H), count);
        Barrier();
        Add(V(PRODUCT_L), V(PRODUCT_L), V(TMP), count);
        Mul(V(TMP), V(X_L), V(COEF_L), count);
        Barrier();
        Add(V(PRODUCT_L), V(PRODUCT_L), V(TMP), count);
    }
    __aicore__ inline void ProductResidual(int32_t part, float coefficient, int32_t count)
    {
        ProductHighResidual(count);
        Muls(V(TMP), V(INPUT_L + part), coefficient, count);
        Barrier();
        Add(V(PRODUCT_L), V(PRODUCT_L), V(TMP), count);
        Barrier();
    }
    __aicore__ inline void ProductVector(int32_t part, LocalTensor<float> coefficient,
        bool negative, int32_t count)
    {
        auto x = V(INPUT_H + part);
        Muls(V(COEF), coefficient, negative ? -1.0f : 1.0f, count);
        Barrier();
        And(V(X_H).template ReinterpretCast<uint16_t>(), x.template ReinterpretCast<uint16_t>(),
            splitMask_.Get<uint16_t>(), count * 2);
        And(V(COEF_H).template ReinterpretCast<uint16_t>(), V(COEF).template ReinterpretCast<uint16_t>(),
            splitMask_.Get<uint16_t>(), count * 2);
        Mul(V(PRODUCT_H), x, V(COEF), count);
        Barrier();
        Sub(V(X_L), x, V(X_H), count);
        Sub(V(COEF_L), V(COEF), V(COEF_H), count);
        Barrier();
        ProductHighResidual(count);
        Mul(V(TMP), V(INPUT_L + part), V(COEF), count);
        Barrier();
        Add(V(PRODUCT_L), V(PRODUCT_L), V(TMP), count);
        Barrier();
    }
    __aicore__ inline void TwoSum(LocalTensor<float> a, LocalTensor<float> b,
        LocalTensor<float> high, LocalTensor<float> low, int32_t count)
    {
        Add(high, a, b, count);
        Barrier();
        Sub(V(VIRTUAL), high, a, count);
        Barrier();
        Sub(V(TMP), high, V(VIRTUAL), count);
        Sub(low, b, V(VIRTUAL), count);
        Barrier();
        Sub(V(TMP), a, V(TMP), count);
        Barrier();
        Add(low, low, V(TMP), count);
        Barrier();
    }
    __aicore__ inline void AddPair(int32_t part, LocalTensor<float> high,
        LocalTensor<float> low, int32_t count)
    {
        TwoSum(V(ACC_H + part), high, V(SUM), V(ERROR), count);
        Add(V(RESIDUAL), V(ACC_L + part), low, count);
        Barrier();
        Add(V(RESIDUAL), V(RESIDUAL), V(ERROR), count);
        Barrier();
        TwoSum(V(SUM), V(RESIDUAL), V(NORMAL_H), V(NORMAL_L), count);
        FiniteMask(V(SUM), count);
        BlendHigh(V(ACC_H + part), V(NORMAL_H), V(SUM), count);
        And(V(ACC_L + part).template ReinterpretCast<uint16_t>(),
            V(NORMAL_L).template ReinterpretCast<uint16_t>(), finiteMask_.Get<uint16_t>(), count * 2);
        Barrier();
    }
    __aicore__ inline bool FiniteScalar(float value)
    {
        return value <= 0x1.fffffep127f && value >= -0x1.fffffep127f;
    }
    __aicore__ inline void BlendHigh(LocalTensor<float> dst, LocalTensor<float> preferred,
        LocalTensor<float> fallback, int32_t count)
    {
        And(V(TMP).template ReinterpretCast<uint16_t>(),
            preferred.template ReinterpretCast<uint16_t>(), finiteMask_.Get<uint16_t>(), count * 2);
        Not(V(ABS_H).template ReinterpretCast<uint16_t>(), finiteMask_.Get<uint16_t>(), count * 2);
        Barrier();
        And(V(ABS_H).template ReinterpretCast<uint16_t>(),
            V(ABS_H).template ReinterpretCast<uint16_t>(), fallback.template ReinterpretCast<uint16_t>(), count * 2);
        Barrier();
        Or(dst.template ReinterpretCast<uint16_t>(), V(TMP).template ReinterpretCast<uint16_t>(),
            V(ABS_H).template ReinterpretCast<uint16_t>(), count * 2);
        Barrier();
    }
    __aicore__ inline void PreserveNonfiniteEstimate(int32_t count)
    {
        FiniteMask(V(ESTIMATE), count);
        if constexpr (Complex) {
            DataCopy(V(COEF).template ReinterpretCast<int32_t>(), finiteMask_.Get<int32_t>(), kTile);
            Barrier();
            FiniteMask(V(ESTIMATE + 1), count);
            And(finiteMask_.Get<uint16_t>(), finiteMask_.Get<uint16_t>(),
                V(COEF).template ReinterpretCast<uint16_t>(), count * 2);
            Barrier();
        }
        for (int32_t part = 0; part < kParts; ++part) {
            BlendHigh(V(ACC_H + part), V(ACC_H + part), V(ESTIMATE + part), count);
            And(V(ACC_L + part).template ReinterpretCast<uint16_t>(),
                V(ACC_L + part).template ReinterpretCast<uint16_t>(), finiteMask_.Get<uint16_t>(), count * 2);
            Barrier();
        }
    }
    __aicore__ inline void ClearNonfiniteLow(int32_t count)
    {
        for (int32_t part = 0; part < kParts; ++part) {
            FiniteMask(V(ACC_H + part), count);
            And(V(ACC_L + part).template ReinterpretCast<uint16_t>(),
                V(ACC_L + part).template ReinterpretCast<uint16_t>(), finiteMask_.Get<uint16_t>(), count * 2);
            Barrier();
        }
    }
    __aicore__ inline void AddProduct(int32_t target, int32_t source, float coefficient, int32_t count)
    {
        Product(source, coefficient, count);
        AddPair(target, V(PRODUCT_H), V(PRODUCT_L), count);
    }
    __aicore__ inline void MultiplyInput(Scalar value, int32_t count)
    {
        Product(0, value.real, count);
        DataCopy(V(ACC_H), V(PRODUCT_H), kTile);
        DataCopy(V(ACC_L), V(PRODUCT_L), kTile);
        Barrier();
        if constexpr (Complex) {
            Product(1, value.real, count);
            DataCopy(V(ACC_H + 1), V(PRODUCT_H), kTile);
            DataCopy(V(ACC_L + 1), V(PRODUCT_L), kTile);
            Barrier();
            AddProduct(0, 1, -value.imag, count);
            AddProduct(1, 0, value.imag, count);
        }
        ClearNonfiniteLow(count);
    }
    __aicore__ inline void SubtractInput(Scalar value, int32_t count)
    {
        AddProduct(0, 0, -value.real, count);
        if constexpr (Complex) {
            AddProduct(0, 1, value.imag, count);
            AddProduct(1, 1, -value.real, count);
            AddProduct(1, 0, -value.imag, count);
        }
    }
    __aicore__ inline Scalar Reciprocal(Scalar value)
    {
        if constexpr (!Complex) { return {1.0f / value.real, 0}; }
        float ar = value.real < 0 ? -value.real : value.real;
        float ai = value.imag < 0 ? -value.imag : value.imag;
        if (ar >= ai) {
            float ratio = value.imag / value.real;
            float real = (1.0f / value.real) / (1.0f + ratio * ratio);
            return {real, -ratio * real};
        }
        float ratio = value.real / value.imag;
        float imag = -(1.0f / value.imag) / (1.0f + ratio * ratio);
        return {-ratio * imag, imag};
    }
    TBuf<TPosition::VECCALC> data_, splitMask_, finiteMask_, offset_, packOffset_;
};
}
#endif
