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

#ifndef CANN_OPS_SPARSE_SPMM_ARCH22_KERNEL_GENERIC_H
#define CANN_OPS_SPARSE_SPMM_ARCH22_KERNEL_GENERIC_H

#include "spmm_kernel_compensated_math.h"

__aicore__ inline float SpmmCombineCompensated(
    float acc, float comp, float alpha, int scale, float betaProduct, float betaError)
{
    // acc 是 A*B 累加和按 2^scale 缩放后的值（真实和 = acc * 2^scale，scale 为
    // SpmmNormalizeProduct 的指数，是 32 的倍数）。把 scale 折进 alpha，避免把 acc
    // 还原回会溢出的真实尺度；若折叠导致 alpha 非有限，则改折进 acc/comp。
    // SpmmProductResidual 内部会再次归一化，保证残差有限。
    float a = alpha;
    float x = acc;
    float c = comp;
    if (scale != 0) {
        const float scaledAlpha = SpmmScaleProduct(alpha, scale);
        if (SpmmFinite(scaledAlpha)) {
            a = scaledAlpha;
        } else {
            x = SpmmScaleProduct(acc, scale);
            c = SpmmScaleProduct(comp, scale);
        }
    }
    // Materialize the FP32 product: contraction would count its residual twice.
    const volatile float scaled = a * x;
    const float scaledComp = a * c;
    const float combined = scaled + betaProduct;
    if (!SpmmFinite(scaled) || !SpmmFinite(scaledComp) ||
        !SpmmFinite(betaProduct) || !SpmmFinite(combined)) {
        return (scaled - scaledComp) + betaProduct;
    }
    // Keep both multiplication and addition residuals until after cancellation.
    const float mulError = SpmmProductResidual(a, x, scaled);
    const float recovered = combined - scaled;
    const float addError = (scaled - (combined - recovered)) + (betaProduct - recovered);
    return combined + (addError + ((mulError - scaledComp) + betaError));
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::SyncScalarToVector()
{
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evt);
    WaitFlag<HardEvent::S_V>(evt);
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::SyncVectorToScalar()
{
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evt);
    WaitFlag<HardEvent::V_S>(evt);
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::SyncScalarToMte3()
{
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::SyncVectorToMte3()
{
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evt);
    WaitFlag<HardEvent::V_MTE3>(evt);
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::SyncMte3ToScalar()
{
    auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(evt);
    WaitFlag<HardEvent::MTE3_S>(evt);
}

template <typename ValT>
__aicore__ inline float SpmmArch22Kernel<ValT>::LoadGeneric(
    GlobalTensor<ValT> &gm, uint64_t index)
{
    if constexpr (kIsFloat) {
        return gm.GetValue(index);
    } else {
        LocalTensor<ValT> raw = genericRawBuf.Get<ValT>();
        LocalTensor<float> fp = genericFloatBuf.Get<float>();
        raw.SetValue(0, gm.GetValue(index));
        SyncScalarToVector();
        Cast<float, ValT>(fp, raw, RoundMode::CAST_NONE, genericScalarElems);
        SyncVectorToScalar();
        return fp.GetValue(0);
    }
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::StoreGeneric(uint64_t index, float value)
{
    LocalTensor<float> fp = genericFloatBuf.Get<float>();
    DataCopyParams copyParams = {1, static_cast<uint16_t>(sizeof(ValT)), 0, 0};
    if constexpr (kIsFloat) {
        fp.SetValue(0, value);
        SyncScalarToMte3();
        DataCopyPad(yGm[index], fp, copyParams);
    } else {
        LocalTensor<ValT> raw = genericRawBuf.Get<ValT>();
        fp.SetValue(0, value);
        SyncScalarToVector();
        Cast<ValT, float>(raw, fp, RoundMode::CAST_ROUND, genericScalarElems);
        SyncVectorToMte3();
        DataCopyPad(yGm[index], raw, copyParams);
    }
    SyncMte3ToScalar();
}

template <typename ValT>
__aicore__ inline uint64_t SpmmArch22Kernel<ValT>::GenericBIndex(
    uint32_t k, uint32_t col) const
{
    const uint32_t row = opB == 0 ? k : col;
    const uint32_t column = opB == 0 ? col : k;
    return bOrder == 0 ? static_cast<uint64_t>(row) * ldb + column :
                         static_cast<uint64_t>(column) * ldb + row;
}

template <typename ValT>
__aicore__ inline uint64_t SpmmArch22Kernel<ValT>::GenericCIndex(
    uint32_t row, uint32_t col) const
{
    return cOrder == 0 ? static_cast<uint64_t>(row) * ldc + col :
                         static_cast<uint64_t>(col) * ldc + row;
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::InitGenericLocalBuffers()
{
    pipe.InitBuffer(genericFloatBuf, genericScalarElems * sizeof(float));
    if constexpr (!kIsFloat) {
        pipe.InitBuffer(genericRawBuf, genericScalarElems * sizeof(ValT));
    }
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::AddGenericProduct(
    float &acc, float &comp, int &scale, float product)
{
    if (!highPrecision) {
        acc += product;
        return;
    }
    const float corrected0 = product - comp;
    float next = acc + corrected0;
    const bool overflow =
        (!SpmmFinite(corrected0) && SpmmFinite(product) && SpmmFinite(comp)) ||
        (!SpmmFinite(next) && SpmmFinite(acc) && SpmmFinite(corrected0));
    if (overflow) {
        // corrected = product - comp 或 next = acc + corrected 溢出（有限操作数得到
        // Inf）：整体下缩 2^32 再重算，scale 同步 +32，保持真实和 = acc * 2^scale 不变。
        acc = SpmmScaleProduct(acc, -32);
        comp = SpmmScaleProduct(comp, -32);
        product = SpmmScaleProduct(product, -32);
        scale += 32;
        const float corrected = product - comp;
        next = acc + corrected;
        comp = (next - acc) - corrected;
        acc = next;
        return;
    }
    comp = (next - acc) - corrected0;
    acc = next;
}

// 计算 value*bValue 并累加进 (acc, comp)。高精度路径下，若中间乘积溢出/下溢
// （value*bValue 为 Inf/NaN，或为非零输入的零），把两操作数归一化到安全量级，
// 用 2^scale 记录真实尺度（product * 2^pExp == value*bValue）。
template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::AddGenericProductScaled(
    float &acc, float &comp, int &scale, float value, float bValue)
{
    if (!highPrecision) {
        AddGenericProduct(acc, comp, scale, alphaVal * value * bValue);
        return;
    }
    float product = value * bValue;
    int pExp = 0;
    // 仅当两操作数均有限时才归一化：非有限输入（±Inf/NaN）直接走 AddGenericProduct
    // 传播 Inf/NaN，与原行为一致；否则 SpmmNormalizeProduct(Inf) 会因 magnitude>up
    // 恒真而死循环挂死 AI Core。
    if (SpmmFinite(value) && SpmmFinite(bValue) &&
        (!SpmmFinite(product) || (product == 0.0f && value != 0.0f && bValue != 0.0f))) {
        const float v = SpmmNormalizeProduct(value, pExp);
        const float b = SpmmNormalizeProduct(bValue, pExp);
        product = v * b;
    }
    // 乘积处于尺度 pExp（真实值 = product * 2^pExp）。与当前累加尺度 scale 对齐：
    // 空累加器直接以乘积尺度初始化；pExp < scale 时把乘积下缩到当前尺度再累加
    // （乘 2^(pExp-scale)，向下缩放安全，仅当真正可忽略时下溢为 0）；pExp > scale
    // 时把累加值下缩到 pExp（安全）。
    if (acc == 0.0f && comp == 0.0f) {
        scale = pExp;
    } else if (pExp < scale) {
        product = SpmmScaleProduct(product, pExp - scale);
    } else if (pExp > scale) {
        acc = SpmmScaleProduct(acc, scale - pExp);
        comp = SpmmScaleProduct(comp, scale - pExp);
        scale = pExp;
    }
    AddGenericProduct(acc, comp, scale, product);
}

template <typename ValT>
__aicore__ inline float SpmmArch22Kernel<ValT>::AccumulateGenericRow(
    int32_t row, uint32_t col, float &comp, int &scale)
{
    float acc = 0.0f;
    comp = 0.0f;
    scale = 0;
    const uint32_t begin =
        static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseOffset);
    const uint32_t end =
        static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseOffset);
    for (uint32_t index = begin; index < end; index++) {
        const uint32_t k = static_cast<uint32_t>(
            colIndicesGm.GetValue(index) - idxBaseOffset);
        const float value = LoadGeneric(valuesGm, index);
        const float bValue = LoadGeneric(bGm, GenericBIndex(k, col));
        AddGenericProductScaled(acc, comp, scale, value, bValue);
    }
    return acc;
}

template <typename ValT>
__aicore__ inline float SpmmArch22Kernel<ValT>::AccumulateGenericColumn(
    int32_t row, uint32_t col, float &comp, int &scale)
{
    float acc = 0.0f;
    comp = 0.0f;
    scale = 0;
    const uint32_t begin =
        static_cast<uint32_t>(transposeRowOffsetsGm.GetValue(row));
    const uint32_t end =
        static_cast<uint32_t>(transposeRowOffsetsGm.GetValue(row + 1));
    for (uint32_t position = begin; position < end; position++) {
        const uint32_t k =
            static_cast<uint32_t>(transposeSourceRowsGm.GetValue(position));
        const uint32_t index =
            static_cast<uint32_t>(transposePermutationGm.GetValue(position));
        const float value = LoadGeneric(valuesGm, index);
        const float bValue = LoadGeneric(bGm, GenericBIndex(k, col));
        AddGenericProductScaled(acc, comp, scale, value, bValue);
    }
    return acc;
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::ApplyBetaGeneric(
    uint64_t out, float &acc, float &betaProduct, float &betaError)
{
    if (skipBeta) {
        return;
    }
    const float cValue = LoadGeneric(cGm, out);
    if (!highPrecision) {
        betaProduct = betaVal * cValue;
        acc += betaProduct;
        return;
    }
    // Residual recovery requires the same explicitly rounded FP32 product.
    const volatile float roundedBeta = betaVal * cValue;
    betaProduct = roundedBeta;
    if (SpmmFinite(betaVal) && SpmmFinite(cValue) && SpmmFinite(betaProduct)) {
        betaError = SpmmProductResidual(betaVal, cValue, betaProduct);
    }
}

template <typename ValT>
__aicore__ inline void SpmmArch22Kernel<ValT>::ProcessGeneric()
{
    for (uint32_t ri = myRowStart; ri < myRowEnd; ri++) {
        const int32_t row = reorderGm.GetValue(ri);
        for (uint32_t col = 0; col < N; col++) {
            float comp = 0.0f;
            int scale = 0;
            float acc = opA == 0 ? AccumulateGenericRow(row, col, comp, scale) :
                                   AccumulateGenericColumn(row, col, comp, scale);
            const uint64_t out = GenericCIndex(static_cast<uint32_t>(row), col);
            float betaProduct = 0.0f;
            float betaError = 0.0f;
            ApplyBetaGeneric(out, acc, betaProduct, betaError);
            StoreGeneric(out, highPrecision ?
                SpmmCombineCompensated(acc, comp, alphaVal, scale, betaProduct, betaError) : acc);
        }
    }
}

#endif
