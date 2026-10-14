/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_SCAN_SUM_H
#define SPSM_ARCH22_SCAN_SUM_H
#include "kernel_operator.h"
namespace spsm_a2 {
// A high/low FP32 pair for each real or interleaved complex component.
// Retain both components through every segmented-scan step and tile carry.
template<int32_t Width> class ScanSum {
    enum Slot { SUM, ERROR, VIRTUAL, TMP, RESIDUAL, HIGH, LOW, FINITE, INVERSE, EXPONENT, SLOTS };
public:
    __aicore__ inline void Init(AscendC::TPipe *pipe)
    {
        pipe->InitBuffer(data_, SLOTS * Width * sizeof(float));
        AscendC::Duplicate(V(EXPONENT).template ReinterpretCast<int32_t>(), 0x7f800000, Width);
        Barrier();
    }
    __aicore__ inline AscendC::LocalTensor<float> High() { return V(HIGH); }
    __aicore__ inline AscendC::LocalTensor<float> Low() { return V(LOW); }
    __aicore__ inline void AddPair(AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> aLow,
        AscendC::LocalTensor<float> b, AscendC::LocalTensor<float> bLow, int32_t count)
    {
        TwoSum(a, b, V(SUM), V(ERROR), count);
        AscendC::Add(V(RESIDUAL), aLow, bLow, count); Barrier();
        AscendC::Add(V(RESIDUAL), V(RESIDUAL), V(ERROR), count); Barrier();
        TwoSum(V(SUM), V(RESIDUAL), V(HIGH), V(LOW), count);
        // Error transforms involve inf-inf for nonfinite sums. Preserve the
        // original IEEE high sum in those lanes instead of introducing NaNs.
        FiniteMask(V(SUM), count);
        BlendHigh(count);
        FiniteMask(V(HIGH), count);
        AscendC::And(V(LOW).template ReinterpretCast<uint16_t>(), V(LOW).template ReinterpretCast<uint16_t>(),
            V(FINITE).template ReinterpretCast<uint16_t>(), count * 2);
        Barrier();
    }
private:
    __aicore__ inline AscendC::LocalTensor<float> V(int32_t slot) { return data_.template Get<float>()[slot * Width]; }
    __aicore__ inline void Barrier() { AscendC::PipeBarrier<PIPE_V>(); }
    __aicore__ inline void TwoSum(AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> b,
        AscendC::LocalTensor<float> high, AscendC::LocalTensor<float> low, int32_t count)
    {
        AscendC::Add(high, a, b, count); Barrier();
        AscendC::Sub(V(VIRTUAL), high, a, count); Barrier();
        AscendC::Sub(V(TMP), high, V(VIRTUAL), count);
        AscendC::Sub(low, b, V(VIRTUAL), count); Barrier();
        AscendC::Sub(V(TMP), a, V(TMP), count); Barrier();
        AscendC::Add(low, low, V(TMP), count); Barrier();
    }
    __aicore__ inline void FiniteMask(AscendC::LocalTensor<float> high, int32_t count)
    {
        AscendC::And(V(FINITE).template ReinterpretCast<uint16_t>(), high.template ReinterpretCast<uint16_t>(),
            V(EXPONENT).template ReinterpretCast<uint16_t>(), count * 2); Barrier();
        AscendC::Adds(V(FINITE).template ReinterpretCast<int32_t>(),
            V(FINITE).template ReinterpretCast<int32_t>(), -0x7f800000, count); Barrier();
        AscendC::ShiftRight(V(FINITE).template ReinterpretCast<int32_t>(),
            V(FINITE).template ReinterpretCast<int32_t>(), 31, count); Barrier();
    }
    __aicore__ inline void BlendHigh(int32_t count)
    {
        auto mask = V(FINITE).template ReinterpretCast<uint16_t>();
        auto inverse = V(INVERSE).template ReinterpretCast<uint16_t>();
        auto tmp = V(TMP).template ReinterpretCast<uint16_t>();
        auto high = V(HIGH).template ReinterpretCast<uint16_t>();
        AscendC::And(high, high, mask, count * 2);
        AscendC::Not(inverse, mask, count * 2); Barrier();
        AscendC::And(tmp, V(SUM).template ReinterpretCast<uint16_t>(), inverse, count * 2); Barrier();
        AscendC::Or(high, high, tmp, count * 2); Barrier();
    }
    AscendC::TBuf<AscendC::TPosition::VECCALC> data_;
};
}
#endif
