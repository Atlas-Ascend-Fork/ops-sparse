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
#pragma once

#include <type_traits>

#include "kernel_operator.h"
#include "spmm_kernel_using.h"

__aicore__ inline void SpmmAdvanceRowCursor(uint32_t gidx, const uint32_t *reArr,
    uint32_t gCnt, uint32_t pitch, uint32_t &gCur, uint32_t &accOff)
{
    while (gCur + 1 < gCnt && gidx >= reArr[gCur]) {
        gCur++;
        accOff += pitch;
    }
}

__aicore__ inline void SpmmIssueFloatAxpy4(LocalTensor<float> &acc,
    LocalTensor<float> &stage, LocalTensor<float> &values, uint32_t valueOff,
    uint32_t accOff, uint32_t stageOff, uint32_t pitch, uint8_t repeats,
    const AscendC::UnaryRepeatParams &repeatParams)
{
    const float v0 = values.GetValue(valueOff);
    const float v1 = values.GetValue(valueOff + 1);
    const float v2 = values.GetValue(valueOff + 2);
    const float v3 = values.GetValue(valueOff + 3);
    Axpy<float, float, false>(acc[accOff], stage[stageOff], v0,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    Axpy<float, float, false>(acc[accOff], stage[stageOff + pitch], v1,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    Axpy<float, float, false>(acc[accOff], stage[stageOff + 2 * pitch], v2,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    Axpy<float, float, false>(acc[accOff], stage[stageOff + 3 * pitch], v3,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
}

__aicore__ inline void SpmmIssueFloatAxpy2(LocalTensor<float> &acc,
    LocalTensor<float> &stage, LocalTensor<float> &values, uint32_t valueOff,
    uint32_t accOff, uint32_t stageOff, uint32_t pitch, uint8_t repeats,
    const AscendC::UnaryRepeatParams &repeatParams)
{
    const float v0 = values.GetValue(valueOff);
    const float v1 = values.GetValue(valueOff + 1);
    Axpy<float, float, false>(acc[accOff], stage[stageOff], v0,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    Axpy<float, float, false>(acc[accOff], stage[stageOff + pitch], v1,
        AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
}

__aicore__ inline void SpmmAxpyFloatRowRuns128(LocalTensor<float> &acc,
    LocalTensor<float> &stage, LocalTensor<float> &values, const uint32_t *reArr,
    uint32_t gCnt, uint32_t batchBase, uint32_t valueBase, uint32_t cnt,
    uint32_t stageBase, uint32_t pitch, uint8_t repeats,
    const AscendC::UnaryRepeatParams &repeatParams, uint32_t &gCur)
{
    uint32_t j = 0;
    uint32_t srcOff = 0;
    uint32_t accOff = gCur * pitch;
    while (j < cnt) {
        const uint32_t gidx = batchBase + j;
        SpmmAdvanceRowCursor(gidx, reArr, gCnt, pitch, gCur, accOff);
        const uint32_t remaining = cnt - j;
        const uint32_t rowCount = reArr[gCur] - gidx;
        const uint32_t runEnd = j + (rowCount < remaining ? rowCount : remaining);
        for (; j + 4 <= runEnd; j += 4) {
            SpmmIssueFloatAxpy4(acc, stage, values, valueBase + j, accOff,
                stageBase + srcOff, pitch, repeats, repeatParams);
            srcOff += 4 * pitch;
        }
        for (; j + 2 <= runEnd; j += 2) {
            SpmmIssueFloatAxpy2(acc, stage, values, valueBase + j, accOff,
                stageBase + srcOff, pitch, repeats, repeatParams);
            srcOff += 2 * pitch;
        }
        for (; j < runEnd; j++) {
            const float val = values.GetValue(valueBase + j);
            Axpy<float, float, false>(acc[accOff], stage[stageBase + srcOff], val,
                AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
            srcOff += pitch;
        }
    }
}

__aicore__ inline void SpmmAxpyFixedFloatSlots(LocalTensor<float> &acc,
    LocalTensor<float> &stage, LocalTensor<float> &values, const uint32_t *reArr,
    uint32_t gCnt, uint32_t batchBase, uint32_t valueBase, uint32_t cnt,
    uint32_t stageBase, uint32_t pitch, int32_t len, uint8_t repeats,
    const AscendC::UnaryRepeatParams &repeatParams, uint32_t &gCur)
{
    if (len == 128) {
        SpmmAxpyFloatRowRuns128(acc, stage, values, reArr, gCnt, batchBase,
            valueBase, cnt, stageBase, pitch, repeats, repeatParams, gCur);
        return;
    }
    uint32_t accOff = gCur * pitch;
    for (uint32_t j = 0; j < cnt; j++) {
        SpmmAdvanceRowCursor(batchBase + j, reArr, gCnt, pitch, gCur, accOff);
        const float val = values.GetValue(valueBase + j);
        Axpy<float, float, false>(acc[accOff], stage[stageBase + j * pitch], val,
            AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    }
}

__aicore__ inline void SpmmAxpyFixedNarrowSlots(LocalTensor<float> &acc,
    LocalTensor<float> &cast, LocalTensor<float> &values, const uint32_t *reArr,
    uint32_t gCnt, uint32_t batchBase, uint32_t valueBase, uint32_t cnt,
    uint32_t pitch, uint8_t repeats, const AscendC::UnaryRepeatParams &repeatParams,
    uint32_t &gCur)
{
    uint32_t accOff = gCur * pitch;
    for (uint32_t j = 0; j < cnt; j++) {
        SpmmAdvanceRowCursor(batchBase + j, reArr, gCnt, pitch, gCur, accOff);
        const float val = values.GetValue(valueBase + j);
        Axpy<float, float, false>(acc[accOff], cast[j * pitch], val,
            AscendC::MASK_PLACEHOLDER, repeats, repeatParams);
    }
}

__aicore__ inline void SpmmAxpyCountFloatSlots(LocalTensor<float> &acc,
    LocalTensor<float> &stage, LocalTensor<float> &values, const uint32_t *reArr,
    uint32_t gCnt, uint32_t batchBase, uint32_t valueBase, uint32_t cnt,
    uint32_t stageBase, uint32_t pitch, int32_t len, float alpha, uint32_t &gCur)
{
    uint32_t accOff = gCur * pitch;
    for (uint32_t j = 0; j < cnt; j++) {
        SpmmAdvanceRowCursor(batchBase + j, reArr, gCnt, pitch, gCur, accOff);
        const float val = values.GetValue(valueBase + j) * alpha;
        Axpy(acc[accOff], stage[stageBase + j * pitch], val, len);
    }
}

__aicore__ inline void SpmmAxpyCountNarrowSlots(LocalTensor<float> &acc,
    LocalTensor<float> &cast, LocalTensor<float> &values, const uint32_t *reArr,
    uint32_t gCnt, uint32_t batchBase, uint32_t valueBase, uint32_t cnt,
    uint32_t pitch, int32_t len, uint32_t &gCur)
{
    uint32_t accOff = gCur * pitch;
    for (uint32_t j = 0; j < cnt; j++) {
        SpmmAdvanceRowCursor(batchBase + j, reArr, gCnt, pitch, gCur, accOff);
        const float val = values.GetValue(valueBase + j);
        Axpy(acc[accOff], cast[j * pitch], val, len);
    }
}

template <typename ValT, typename PipeCtxT>
__aicore__ inline void SpmmAxpySegmentSlots(LocalTensor<float> &acc, PipeCtxT &pc,
    const uint32_t *reArr, uint32_t gCnt, uint32_t segBase, uint32_t ubIdx,
    uint32_t cnt, uint32_t sel, uint32_t pitch, uint32_t stageStride,
    float alpha, uint32_t &gCur)
{
    const int32_t len = static_cast<int32_t>(pc.chunkAlign);
    const uint32_t stageOff = sel * stageStride;
    constexpr bool isFloat = std::is_same_v<ValT, float>;
    const bool fixedMask = len > 0 && len % 64 == 0 && len / 64 <= 255;
    if (fixedMask && (!isFloat || alpha == 1.0f)) {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(0), static_cast<uint64_t>(-1));
        const uint8_t repeats = static_cast<uint8_t>(len / 64);
        const AscendC::UnaryRepeatParams repeatParams{1, 1, 8, 8};
        if constexpr (std::is_same_v<ValT, float>) {
            SpmmAxpyFixedFloatSlots(acc, pc.stage, pc.valUb, reArr, gCnt,
                segBase + ubIdx, ubIdx, cnt, stageOff, pitch, len, repeats,
                repeatParams, gCur);
        } else {
            SpmmAxpyFixedNarrowSlots(acc, pc.cast, pc.valF, reArr, gCnt,
                segBase + ubIdx, ubIdx, cnt, pitch, repeats, repeatParams, gCur);
        }
        return;
    }
    if constexpr (std::is_same_v<ValT, float>) {
        SpmmAxpyCountFloatSlots(acc, pc.stage, pc.valUb, reArr, gCnt,
            segBase + ubIdx, ubIdx, cnt, stageOff, pitch, len, alpha, gCur);
    } else {
        SpmmAxpyCountNarrowSlots(acc, pc.cast, pc.valF, reArr, gCnt,
            segBase + ubIdx, ubIdx, cnt, pitch, len, gCur);
    }
}
