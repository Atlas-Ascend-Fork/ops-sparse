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
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "sddmm_simd_kernel.h"
#include "sddmm_complex_kernel.h"
#include "sddmm_transform_kernel.h"
#include "sddmm_bsr_small_kernel.h"
#include "../host/sddmm_launch.h"
#include "../host/sddmm_tiling_abi.h"

// Shared complex indexing constants are defined by sddmm_complex_kernel.h.
// 把两次实数 SDDMM 的结果合成回 complex64，顺带施加 alpha / beta。
//
// 输入 P 来自 Xr 那一遍、Q 来自 Xi 那一遍，槽位 2e 是第 e 个非零元的第 2j 个
// 实数（Y 的实部），槽位 2e+1 是第 2j+1 个（虚部）。于是
//
//   P[2e] = A = sum xr*yr   P[2e+1] = C = sum xr*yi
//   Q[2e] = B = sum xi*yr   Q[2e+1] = D = sum xi*yi
//   Re = A - D              Im = C + B
//
// 令 Qb = swap(Q)，直接计算 Z = P + parity_signed * Qb。
// 偶数位得到实部差，奇数位得到虚部和，避免无用中间差值溢出。
//
// 复数乘 alpha 在交错布局下是 out = ar*Z + parity_signed * swap(ai*Z)，其中
// parity_signed 偶数位 -1、奇数位 +1；beta 那项同理，只是不换符号：
// out = ar*Z + br*old + parity_signed * swap(ai*Z + bi*old)。整个 kernel 只有
// 两次 gather（都是固定的相邻对调），其余全是逐元素向量指令。
class SddmmComplexCombine {
public:
    __aicore__ inline SddmmComplexCombine() {}

    __aicore__ inline void Init(GM_ADDR prodP, GM_ADDR prodQ, GM_ADDR values, uint32_t slots, float alphaReal,
                                float alphaImag, float betaReal, float betaImag, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        total = slots;
        alphaR = alphaReal;
        alphaI = alphaImag;
        betaR = betaReal;
        betaI = betaImag;
        const uint64_t elems = static_cast<uint64_t>(slots);
        prodPGm.SetGlobalBuffer((__gm__ float*)prodP, elems);
        prodQGm.SetGlobalBuffer((__gm__ float*)prodQ, elems);
        valuesGm.SetGlobalBuffer((__gm__ float*)values, elems);
        pipe->InitBuffer(inQueueP, 1, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(inQueueQ, 1, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(inQueueC, 1, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(outQueue, 1, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(qbBuf, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(sBuf, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(zBuf, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(wBuf, kCplxDotChunk * sizeof(float));
        pipe->InitBuffer(tmpBuf, kCplxDotChunk * sizeof(float));
        // swapOff / half / tmp / parity（int32）+ sign（float），五个等长表
        pipe->InitBuffer(tableBuf, 5u * kCplxDotChunk * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (total == 0) {
            return;
        }
        BuildTables();
        // 分块必须从偶数槽位开始，否则奇偶表和全局奇偶对不上。
        const uint32_t chunks = (total + kCplxDotChunk - 1u) / kCplxDotChunk;
        uint32_t first = 0;
        uint32_t last = 0;
        CplxSlice(chunks, first, last);
        for (uint32_t c = first; c < last; ++c) {
            DoChunk(c);
        }
    }

private:
    __aicore__ inline void BuildTables()
    {
        AscendC::LocalTensor<int32_t> tab = tableBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> swapOff = tab;
        AscendC::LocalTensor<int32_t> half = tab[kCplxDotChunk];
        AscendC::LocalTensor<int32_t> tmp = tab[2u * kCplxDotChunk];
        AscendC::LocalTensor<int32_t> parity = tab[3u * kCplxDotChunk];
        AscendC::LocalTensor<float> alt = half.ReinterpretCast<float>();
        AscendC::LocalTensor<float> sign = tab[4u * kCplxDotChunk].ReinterpretCast<float>();

        const int32_t cap = static_cast<int32_t>(kCplxDotChunk);
        // swapOff[i] = 4 * (i ^ 1)，即相邻槽位的字节偏移。arch22 上没有 Ands，
        // And 又只吃 16 位整数，所以 i^1 按 i + 1 - 2*(i&1) 算；i&1 仍靠右移
        // 求 floor 再减回去（非负数的算术右移就是 floor）。
        AscendC::CreateVecIndex<int32_t>(swapOff, 0, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight<int32_t>(half, swapOff, 1, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp, half, 2, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(parity, swapOff, tmp, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(swapOff, swapOff, 4, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(swapOff, swapOff, 4, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp, parity, 8, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(swapOff, swapOff, tmp, cap);
        AscendC::PipeBarrier<PIPE_V>();

        // alt = 位置奇偶（0/1 的浮点）；half 这张 int32 表到这里已经用完，借它的
        // 寄存器空间放浮点结果即可（两者同为 4*cap 字节）。
        AscendC::Cast<float, int32_t>(alt, parity, AscendC::RoundMode::CAST_RINT, cap);
        AscendC::PipeBarrier<PIPE_V>();
        // sign = 2*parity - 1：偶数位 -1、奇数位 +1。
        AscendC::Muls(sign, alt, 2.0f, cap);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sign, sign, -1.0f, cap);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline AscendC::LocalTensor<float> Pull(AscendC::TQue<AscendC::TPosition::VECIN, 1>& q,
                                                       AscendC::GlobalTensor<float>& gm, uint32_t off, uint32_t cnt)
    {
        AscendC::LocalTensor<float> buf = q.AllocTensor<float>();
        AscendC::DataCopyExtParams p{1, cnt * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(buf, gm[off], p, pad);
        q.EnQue(buf);
        return q.DeQue<float>();
    }

    __aicore__ inline AscendC::LocalTensor<float> PullValues(uint32_t base, uint32_t cnt)
    {
        AscendC::LocalTensor<float> old = inQueueC.AllocTensor<float>();
        if (betaR == 0.0f && betaI == 0.0f) {
            // C 只被乘以零。不去碰调用方的缓冲区，既省一趟搬运，也避免把未初始化
            // 的位型（可能是 NaN）带进来。
            AscendC::Duplicate(old, 0.0f, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopyExtParams cp{1, cnt * (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> cpd{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(old, valuesGm[base], cp, cpd);
        }
        inQueueC.EnQue(old);
        old = inQueueC.DeQue<float>();

        return old;
    }

    __aicore__ inline void ScaleTerms(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> z,
                                      AscendC::LocalTensor<float> old, float a, float b, uint32_t cnt)
    {
        if (a == 0.0f) {
            AscendC::Duplicate(dst, 0.0f, static_cast<int32_t>(cnt));
        } else {
            AscendC::Muls(dst, z, a, static_cast<int32_t>(cnt));
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (b != 0.0f) {
            AscendC::Axpy<float, float>(dst, old, b, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline AscendC::LocalTensor<float> ApplyScalars(AscendC::LocalTensor<float> z,
                                                               AscendC::LocalTensor<float> old, uint32_t cnt)
    {
        auto tab = tableBuf.Get<int32_t>();
        auto swapOff = tab.ReinterpretCast<uint32_t>();
        auto sign = tab[4u * kCplxDotChunk].ReinterpretCast<float>();
        auto s = sBuf.Get<float>();
        auto tmp = tmpBuf.Get<float>();
        // u = alphaR * Z + betaR * C
        ScaleTerms(s, z, old, alphaR, betaR, cnt);
        // w = alphaI * Z + betaI * C，随后连同符号一起做相邻对调。
        AscendC::LocalTensor<float> w = wBuf.Get<float>();
        ScaleTerms(w, z, old, alphaI, betaI, cnt);
        AscendC::LocalTensor<float> wb = tmp;
        AscendC::Gather<float>(wb, w, swapOff, 0, cnt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(wb, wb, sign, static_cast<int32_t>(cnt));
        AscendC::PipeBarrier<PIPE_V>();
        inQueueC.FreeTensor(old);

        return wb;
    }

    __aicore__ inline void DoChunk(uint32_t chunk)
    {
        const uint32_t base = chunk * kCplxDotChunk;
        const uint32_t cnt = CplxMinU32(total - base, kCplxDotChunk);

        AscendC::LocalTensor<int32_t> tab = tableBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> swapOff = tab.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<float> sign = tab[4u * kCplxDotChunk].ReinterpretCast<float>();

        AscendC::LocalTensor<float> p = Pull(inQueueP, prodPGm, base, cnt);
        AscendC::LocalTensor<float> q = Pull(inQueueQ, prodQGm, base, cnt);

        auto old = PullValues(base, cnt);

        AscendC::LocalTensor<float> qb = qbBuf.Get<float>();
        AscendC::Gather<float>(qb, q, swapOff, 0, cnt);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> s = sBuf.Get<float>();
        AscendC::Mul(qb, qb, sign, static_cast<int32_t>(cnt));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<float> z = zBuf.Get<float>();
        AscendC::Add(z, p, qb, static_cast<int32_t>(cnt));
        AscendC::PipeBarrier<PIPE_V>();
        inQueueP.FreeTensor(p);
        inQueueQ.FreeTensor(q);

        auto wb = ApplyScalars(z, old, cnt);

        AscendC::LocalTensor<float> out = outQueue.AllocTensor<float>();
        AscendC::Add(out, s, wb, static_cast<int32_t>(cnt));
        AscendC::PipeBarrier<PIPE_V>();
        outQueue.EnQue(out);
        out = outQueue.DeQue<float>();
        AscendC::DataCopyExtParams op{1, cnt * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(valuesGm[base], out, op);
        outQueue.FreeTensor(out);
    }

    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueP;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueC;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qbBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tableBuf;
    AscendC::GlobalTensor<float> prodPGm;
    AscendC::GlobalTensor<float> prodQGm;
    AscendC::GlobalTensor<float> valuesGm;
    float alphaR = 1.0f;
    float alphaI = 0.0f;
    float betaR = 0.0f;
    float betaI = 0.0f;
    uint32_t total = 0;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2201)
#error "SDDMM arch22: this TU is only for dav-2201 / Ascend 910B (__NPU_ARCH__==2201)."
#endif

// One entry point per (X/Y, C) storage pair of the support matrix. Compute
// always accumulates in FP32, so the pair alone picks the instantiation; a pair
// listed here but absent from the host validator (or vice versa) is a bug.
__aicore__ inline bool SddmmPatternAllowsWrite(const SddmmKernelTilingData& td, AscendC::TPipe& pipe)
{
    if (td.patternFlags == 0) {
        return true;
    }
    // The preceding full comparison kernel has completed on this stream.
    // Flags are 0/1 and every flag word is initialized, including idle cores.
    const uint32_t words = td.patternFlagWords;
    AscendC::TBuf<AscendC::TPosition::VECCALC> check;
    pipe.InitBuffer(check, (2 * words + 8) * sizeof(float));
    auto raw = check.Get<int32_t>();
    auto errors = raw[words].ReinterpretCast<float>();
    auto reduced = raw[2 * words].ReinterpretCast<float>();
    AscendC::GlobalTensor<int32_t> flags;
    flags.SetGlobalBuffer((__gm__ int32_t*)td.patternFlags);
    AscendC::DataCopy(raw, flags, words);
    auto mte = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte);
    AscendC::Cast(errors, raw, AscendC::RoundMode::CAST_NONE, words);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(reduced, errors, raw.ReinterpretCast<float>(), words, false);
    auto scalar = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(scalar);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(scalar);
    // ReduceMax leaves a partial mask. Restore it before either entry path exits.
    AscendC::ResetMask();
    AscendC::SetMaskNorm();
    return reduced.GetValue(0) == 0.0f;
}

#define SDDMM_DEFINE_ENTRY(Name, XT, CT)                                                                               \
    extern "C" __global__ __aicore__ void Name(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,                      \
                                               GM_ADDR csrColIndices, GM_ADDR csrValues, SddmmKernelTilingData tiling) \
    {                                                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                                \
        AscendC::TPipe pipe;                                                                                           \
        if (!SddmmPatternAllowsWrite(tiling, pipe)) {                                                                  \
            return;                                                                                                    \
        }                                                                                                              \
        SddmmKernel<XT, CT> op;                                                                                        \
        op.Init(matA, matB, csrRowOffsets, csrColIndices, csrValues, tiling, &pipe);                                   \
        op.Process();                                                                                                  \
    }

SDDMM_DEFINE_ENTRY(sddmm_kernel_f32, float, float)
SDDMM_DEFINE_ENTRY(sddmm_kernel_f16, half, half)
SDDMM_DEFINE_ENTRY(sddmm_kernel_half_f32, half, float)
SDDMM_DEFINE_ENTRY(sddmm_kernel_bf16, bfloat16_t, bfloat16_t)
SDDMM_DEFINE_ENTRY(sddmm_kernel_bf16_f32, bfloat16_t, float)

template <typename Word, uint32_t Lanes>
__global__ __aicore__ void sddmm_bsr_repack(GM_ADDR blockValues, GM_ADDR scalarValues, GM_ADDR rowPtr, uint32_t rows,
                                            uint32_t blocks, uint32_t bd, int32_t base, bool colMajor, bool unpack)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmBsrRepack<Word, Lanes> op;
    op.Init(blockValues, scalarValues, rowPtr, rows, blocks, bd, base, colMajor, unpack, &pipe);
    op.Process();
}

// Dispatches to the instantiation matching `tiling`'s dtype pair. The host
// validator admits exactly the pairs listed here; anything else is rejected.

// ---------------------------------------------------------------------------
// B-operand transpose entry points
// ---------------------------------------------------------------------------

// Yt[j * pitch + k] = Y[k * ldy + j], one entry point per storage dtype so that
// the Gather and the two DataCopyPad calls see real element types.
#define SDDMM_DEFINE_TRANSPOSE_ENTRY(Name, XT)                                                                    \
    extern "C" __global__ __aicore__ void Name(GM_ADDR y, GM_ADDR yt, uint32_t rows, uint32_t cols, uint32_t yLd, \
                                               uint32_t pitch)                                                    \
    {                                                                                                             \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                           \
        AscendC::TPipe pipe;                                                                                      \
        SddmmYTranspose<XT> op;                                                                                   \
        op.Init(y, yt, rows, cols, yLd, pitch, &pipe);                                                            \
        op.Process();                                                                                             \
        /* Gather leaves a partial mask; restore it even when Process returns early. */                           \
        AscendC::ResetMask();                                                                                     \
        AscendC::SetMaskNorm();                                                                                   \
    }

SDDMM_DEFINE_TRANSPOSE_ENTRY(sddmm_y_transpose_f32, float)
SDDMM_DEFINE_TRANSPOSE_ENTRY(sddmm_y_transpose_f16, half)
SDDMM_DEFINE_TRANSPOSE_ENTRY(sddmm_y_transpose_bf16, bfloat16_t)

// ---------------------------------------------------------------------------
// complex64 entry points
// ---------------------------------------------------------------------------

// The three helpers are small and their argument lists differ; spelling each
// entry point out is shorter than a macro that would need a variadic forward.
extern "C" __global__ __aicore__ void sddmm_complex_split_x(GM_ADDR x, GM_ADDR xR, GM_ADDR xI, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmComplexSplitX helper;
    helper.Init(x, xR, xI, count, &pipe);
    helper.Process();
}

extern "C" __global__ __aicore__ void sddmm_complex_prep_csr(GM_ADDR rowPtr, GM_ADDR colIdx, GM_ADDR rowPtr2,
                                                             GM_ADDR colIdx2, uint32_t rows, uint32_t nnz,
                                                             int32_t idxBase)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmComplexPrepCsr helper;
    helper.Init(rowPtr, colIdx, rowPtr2, colIdx2, rows, nnz, idxBase, &pipe);
    helper.Process();
}

extern "C" __global__ __aicore__ void sddmm_complex_combine(GM_ADDR prodP, GM_ADDR prodQ, GM_ADDR values,
                                                            uint32_t slots, float alphaReal, float alphaImag,
                                                            float betaReal, float betaImag)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmComplexCombine helper;
    helper.Init(prodP, prodQ, values, slots, alphaReal, alphaImag, betaReal, betaImag, &pipe);
    helper.Process();
}

extern "C" __global__ __aicore__ void sddmm_complex_fused_products(GM_ADDR xr, GM_ADDR xi, GM_ADDR yt, GM_ADDR rp,
                                                                   GM_ADDR ci, GM_ADDR values, uint32_t rows,
                                                                   uint32_t cols, uint32_t k, uint32_t pitch,
                                                                   uint32_t nnz, int32_t base, float ar, float ai,
                                                                   float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    SddmmComplexFusedProducts op;
    op.Init(xr, xi, yt, rp, ci, values, rows, cols, k, pitch, nnz, base, ar, ai, br, bi, &pipe);
    op.Process();
}

extern "C" aclsparseStatus_t sddmm_bsr_repack_launch(GM_ADDR blockValues, GM_ADDR scalarValues, GM_ADDR rowPtr,
                                                     uint32_t rows, uint32_t blocks, uint32_t bd, int32_t base,
                                                     aclDataType dtype, bool colMajor, bool unpack, uint32_t cores,
                                                     void* stream)
{
    if (blocks == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        sddmm_bsr_repack<uint16_t, 1>
            <<<cores, nullptr, stream>>>(blockValues, scalarValues, rowPtr, rows, blocks, bd, base, colMajor, unpack);
    } else if (dtype == ACL_FLOAT) {
        sddmm_bsr_repack<float, 1>
            <<<cores, nullptr, stream>>>(blockValues, scalarValues, rowPtr, rows, blocks, bd, base, colMajor, unpack);
    } else if (dtype == ACL_COMPLEX64) {
        sddmm_bsr_repack<float, 2>
            <<<cores, nullptr, stream>>>(blockValues, scalarValues, rowPtr, rows, blocks, bd, base, colMajor, unpack);
    } else {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_kernel_launch(GM_ADDR matA, GM_ADDR matB, GM_ADDR csrRowOffsets,
                                                 GM_ADDR csrColIndices, GM_ADDR csrValues,
                                                 const SddmmTilingData& tiling, uint32_t numBlocks, void* stream)
{
#define SDDMM_TRY_ENTRY(Name, XT, CT)                                                             \
    if (tiling.xType == SDDMM_DTYPE_##XT && tiling.cType == SDDMM_DTYPE_##CT) {                   \
        Name<<<numBlocks, nullptr, stream>>>(matA, matB, csrRowOffsets, csrColIndices, csrValues, \
                                             SddmmSerializeTiling(tiling));                       \
        return ACL_SPARSE_STATUS_SUCCESS;                                                         \
    }

    SDDMM_TRY_ENTRY(sddmm_kernel_f32, FP32, FP32)
    SDDMM_TRY_ENTRY(sddmm_kernel_f16, FP16, FP16)
    SDDMM_TRY_ENTRY(sddmm_kernel_half_f32, FP16, FP32)
    SDDMM_TRY_ENTRY(sddmm_kernel_bf16, BF16, BF16)
    SDDMM_TRY_ENTRY(sddmm_kernel_bf16_f32, BF16, FP32)

#undef SDDMM_TRY_ENTRY
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

extern "C" aclsparseStatus_t sddmm_y_transpose_launch(GM_ADDR y, GM_ADDR yt, uint32_t rows, uint32_t cols, uint32_t yLd,
                                                      uint32_t pitch, uint32_t xType, uint32_t numBlocks, void* stream)
{
    if (xType == SDDMM_DTYPE_FP32) {
        sddmm_y_transpose_f32<<<numBlocks, nullptr, stream>>>(y, yt, rows, cols, yLd, pitch);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (xType == SDDMM_DTYPE_FP16) {
        sddmm_y_transpose_f16<<<numBlocks, nullptr, stream>>>(y, yt, rows, cols, yLd, pitch);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (xType == SDDMM_DTYPE_BF16) {
        sddmm_y_transpose_bf16<<<numBlocks, nullptr, stream>>>(y, yt, rows, cols, yLd, pitch);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

extern "C" aclsparseStatus_t sddmm_complex_launch(GM_ADDR matX, GM_ADDR matY, GM_ADDR csrRowOffsets,
                                                  GM_ADDR csrColIndices, GM_ADDR csrValues, uint32_t rows, uint32_t k,
                                                  uint32_t nnz, uint32_t idxBase, float alphaReal, float alphaImag,
                                                  float betaReal, float betaImag, const SddmmTilingData& innerTiling,
                                                  void* workspace, void* stream)
{
    (void)matX;
    (void)csrRowOffsets;
    (void)csrColIndices;
    (void)idxBase;
    if (workspace == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint32_t cols = innerTiling.matBCols / 2u;
    const SddmmComplexPlan plan = SddmmComplexPlanOf(rows, k, cols, nnz);
    uint8_t* base = static_cast<uint8_t*>(workspace);
    GM_ADDR matYArg = plan.useYt ? base + plan.yTrans : matY;

    if (plan.directOutput) {
        sddmm_complex_fused_products<<<innerTiling.blockDim, nullptr, stream>>>(
            base + plan.xReal, base + plan.xImag, matYArg, csrRowOffsets, csrColIndices, csrValues, rows, cols, k,
            plan.yPitch, nnz, static_cast<int32_t>(idxBase), alphaReal, alphaImag, betaReal, betaImag);
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    // Two ordinary fp32 SDDMM passes over the doubled pattern. Both read the same
    // expanded structure; only the X plane differs. beta is zero here because the
    // caller's C is folded in by the combine pass, which is also where the complex
    // alpha/beta are applied -- so the real kernel's beta==0 path must not touch
    // P/Q (it does not: the beta guard skips the C read entirely).
    const aclsparseStatus_t first = sddmm_kernel_launch(base + plan.xReal, matYArg, base + plan.rowPtr,
                                                        base + plan.colIdx, base + plan.prodP, innerTiling,
                                                        innerTiling.blockDim, stream);
    if (first != ACL_SPARSE_STATUS_SUCCESS) {
        return first;
    }
    const aclsparseStatus_t second = sddmm_kernel_launch(base + plan.xImag, matYArg, base + plan.rowPtr,
                                                         base + plan.colIdx, base + plan.prodQ, innerTiling,
                                                         innerTiling.blockDim, stream);
    if (second != ACL_SPARSE_STATUS_SUCCESS) {
        return second;
    }

    sddmm_complex_combine<<<innerTiling.blockDim, nullptr, stream>>>(
        base + plan.prodP, base + plan.prodQ, csrValues, 2u * nnz, alphaReal, alphaImag, betaReal, betaImag);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_complex_prepare(GM_ADDR matX, GM_ADDR matY, GM_ADDR csrRowOffsets,
                                                   GM_ADDR csrColIndices, uint32_t rows, uint32_t k, uint32_t cols,
                                                   uint32_t nnz, uint32_t idxBase, uint32_t blocks, void* workspace,
                                                   void* stream)
{
    if (workspace == nullptr || blocks == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const SddmmComplexPlan plan = SddmmComplexPlanOf(rows, k, cols, nnz);
    uint8_t* base = static_cast<uint8_t*>(workspace);

    // X is m x k and the pattern is at most nnz entries, so every core that gets
    // work always finds some; the slice helper leaves the rest idle.
    sddmm_complex_split_x<<<blocks, nullptr, stream>>>(matX, base + plan.xReal, base + plan.xImag, rows * k);
    if (!plan.directOutput) {
        sddmm_complex_prep_csr<<<blocks, nullptr, stream>>>(csrRowOffsets, csrColIndices, base + plan.rowPtr,
                                                            base + plan.colIdx, rows, nnz,
                                                            static_cast<int32_t>(idxBase));
    }
    if (plan.useYt) {
        // A complex64 row is an interleaved pair of fp32 values, so the flat
        // view of Y is already [K, 2N]. Transpose it once and let both real
        // SDDMM passes share the packed columns.
        return sddmm_y_transpose_launch(matY, base + plan.yTrans, k, 2u * cols, 2u * cols, plan.yPitch,
                                        SDDMM_DTYPE_FP32, blocks, stream);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

namespace {
template <typename T>
aclsparseStatus_t LaunchSmall(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR cols, GM_ADDR values, uint32_t m, uint32_t n,
                              uint32_t k, uint32_t bd, uint32_t blocks, int32_t base, bool columnMajor,
                              aclDataType cType, float alpha, float beta, void* stream)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    const uint32_t cores = std::min(blocks, platform->GetCoreNumAiv());
    if (!cores) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    if (cType == ACL_FLOAT) {
        BsrSmall<T, float>
            <<<cores, nullptr, stream>>>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, alpha, beta);
    } else {
        BsrSmall<T, T>
            <<<cores, nullptr, stream>>>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, alpha, beta);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

extern "C" aclsparseStatus_t sddmm_bsr_complex_vector_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR cols,
                                                             GM_ADDR values, uint32_t m, uint32_t n, uint32_t k,
                                                             uint32_t bd, uint32_t blocks, int32_t base,
                                                             bool columnMajor, float ar, float ai, float br, float bi,
                                                             void* stream)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    const uint32_t width = std::min(bd, bd >= 16 ? 32u : 16u);
    const uint32_t rowsPerTile = bd >= 16 ? 2 : 4;
    if (bd == 0 || width == 0 || rowsPerTile == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const uint32_t cores = std::min(blocks * (bd / rowsPerTile) * (bd / width), platform->GetCoreNumAiv());
    if (!cores) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    BsrComplexVector<<<cores, nullptr, stream>>>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, ar,
                                                 ai, br, bi);
    return ACL_SPARSE_STATUS_SUCCESS;
}

extern "C" aclsparseStatus_t sddmm_bsr_small_launch(GM_ADDR x, GM_ADDR y, GM_ADDR rows, GM_ADDR cols, GM_ADDR values,
                                                    uint32_t m, uint32_t n, uint32_t k, uint32_t bd, uint32_t blocks,
                                                    int32_t base, bool columnMajor, aclDataType xType,
                                                    aclDataType cType, float alpha, float beta, void* stream)
{
    if (xType == ACL_FLOAT16) {
        return LaunchSmall<half>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, cType, alpha, beta,
                                 stream);
    }
    if (xType == ACL_BF16) {
        return LaunchSmall<bfloat16_t>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, cType, alpha,
                                       beta, stream);
    }
    if (xType == ACL_FLOAT) {
        return LaunchSmall<float>(x, y, rows, cols, values, m, n, k, bd, blocks, base, columnMajor, cType, alpha, beta,
                                  stream);
    }
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}
