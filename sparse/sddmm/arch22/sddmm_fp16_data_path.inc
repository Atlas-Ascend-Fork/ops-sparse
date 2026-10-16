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

// Private class-body implementation fragment for KernelSparseSddmmFp16.
//
// This file intentionally has the .inc suffix and no include guard: it is
// included exactly once, inside the private section of sddmm_fp16_kernel.h.
// The methods require that class's UB buffers and runtime state, so making it
// a standalone public header would be misleading.
__aicore__ inline void Gather16(AscendC::LocalTensor<float> &ub,
                                AscendC::GlobalTensor<half> &gm,
                                uint64_t off, uint32_t stride, uint32_t len,
                                bool protectReuse = true) {
    if (stride == 1u && len > 0) {
        AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, static_cast<half>(0.0f)};
        AscendC::DataCopyPad(hBuf, gm[off], copyParams, padParams);
        SyncMte2ToV();
        AscendC::Cast<float, half>(ub, hBuf, AscendC::RoundMode::CAST_NONE, len);
        if (protectReuse) {
            SyncVToMte2();
        }
        return;
    }
    AscendC::LocalTensor<half> pad32 = stridePadBuf.Get<half>();
    AscendC::LocalTensor<half> hBuf = halfBuf.Get<half>();
    AscendC::LocalTensor<int32_t> offsets = strideOffsetBuf.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> offsetsU = offsets.ReinterpretCast<uint32_t>();
    const uint32_t srcGapBytes = (stride - 1u) * static_cast<uint32_t>(sizeof(half));
    for (uint32_t base = 0; base < len; base += kFp16StrideChunk) {
        const uint32_t count = MinU32(len - base, kFp16StrideChunk);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(count), static_cast<uint32_t>(sizeof(half)),
                                          srcGapBytes, 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, static_cast<half>(0.0f)};
        AscendC::DataCopyPad(pad32, gm[off + static_cast<uint64_t>(base) * stride], params, pad);
        SyncMte2ToV();
        AscendC::Gather<half>(hBuf[base], pad32, offsetsU, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (base + count < len) {
            SyncVToMte2();
        }
    }
    AscendC::Cast<float, half>(ub, hBuf, AscendC::RoundMode::CAST_NONE, len);
    if (protectReuse) {
        SyncVToMte2();
    }
}

__aicore__ inline float DotProduct(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                   uint32_t bStride, int aCache) {
    return K <= kBlockLen ? DotProductSingle(aOff, aStride, bOff, bStride, aCache)
                           : DotProductScalar(aOff, aStride, bOff, bStride);
}

__aicore__ inline void ReduceDot16(AscendC::LocalTensor<float> &dst,
                                   AscendC::LocalTensor<float> &product,
                                   AscendC::LocalTensor<float> &temporary) {
    if (K <= kElemPerRep) {
        AscendC::WholeReduceSum<float>(dst, product, static_cast<int32_t>(K), 1, 1, 1, 8);
    } else {
        AscendC::ReduceSum<float, true>(dst, product, temporary, static_cast<int32_t>(K));
    }
}

__aicore__ inline float DotProductSingle(uint64_t aOff, uint32_t aStride, uint64_t bOff,
                                         uint32_t bStride, int aCache) {
    AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
    AscendC::LocalTensor<float> b = bColBuf.Get<float>();
    AscendC::LocalTensor<float> product = mulBuf.Get<float>();
    AscendC::LocalTensor<float> temporary = reduceTmpBuf.Get<float>();
    AscendC::LocalTensor<float> result = reduceDstBuf.Get<float>();
    if (aCache == kACacheNone) {
        Gather16(a, matAGm, aOff, aStride, K);
    }
    Gather16(b, matBGm, bOff, bStride, K, false);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(product, a, b, K);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceDot16(result, product, temporary);
    SyncVToS();
    return result.GetValue(0);
}

__aicore__ inline float DotProductScalar(uint64_t aOff, uint32_t aStride,
                                         uint64_t bOff, uint32_t bStride) {
    AscendC::LocalTensor<float> a = aRowBuf.Get<float>();
    AscendC::LocalTensor<float> b = bColBuf.Get<float>();
    float acc = 0.0f;
    for (uint32_t base = 0; base < K; base += kBlockLen) {
        const uint32_t length = MinU32(K - base, kBlockLen);
        Gather16(a, matAGm, aOff + static_cast<uint64_t>(base) * aStride, aStride, length);
        Gather16(b, matBGm, bOff + static_cast<uint64_t>(base) * bStride, bStride, length, false);
        SyncVToS();
        for (uint32_t i = 0; i < length; ++i) {
            acc += a.GetValue(i) * b.GetValue(i);
        }
    }
    return acc;
}

__aicore__ inline void GatherFull16(AscendC::LocalTensor<float> &dst,
                                    AscendC::GlobalTensor<half> &gm,
                                    uint64_t off, uint32_t stride, uint32_t len) {
    for (uint32_t base = 0; base < len; base += kBlockLen) {
        const uint32_t chunk = MinU32(len - base, kBlockLen);
        AscendC::LocalTensor<float> slice = dst[base];
        Gather16(slice, gm, off + static_cast<uint64_t>(base) * stride, stride, chunk);
    }
}
