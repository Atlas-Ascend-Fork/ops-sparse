/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file matmul_epilogue_kernel.cpp
 * \brief sparseLt 非融合 epilogue kernel（AIV_ONLY，splitK>1 路径）+ 启动器。
 * 超大源文件整改拆分单元（原 matmul_kernel.cpp，见
 * matmul_kernel.cpp 文件头）；共享基础设施位于 matmul_kernel_common.h。
 * kernel：
 *   - epilogue_kernel (AIV_ONLY, 低层 AscendC Vector API): D = alpha*reduce(temp) + beta*C
 * Tiling 经 splt_load_tiling(tilingGm) 从 GM 加载（shared/aclsparselt_internal.h）。
 */


#include <cstdint>

#include "log/log.h"

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "shared/aclsparselt_internal.h"
#include "matmul_kernel.h"
#include "matmul_kernel_common.h"

// ============================================================================
// EPILOGUE KERNEL（AIV_ONLY）
//   D = alpha * reduce_k(temp) + beta * C
//   temp 为 FP32（splitK 部分和）；C/D 为 T（FP16 或 FP32）。
//   以 DataCopyPad（GM<->UB）+ UB 内标量 GetValue/SetValue 取代原来的
//   标量 __gm__ 访问。标量 GM 访问（__gm__ T* 下标）在
//   API 黑名单中（ascendc-api-best-practices），在 AIV 核上会静默丢数据
//   —— temp 值正确但 D 全为零。
// ============================================================================

// splitK epilogue 的 INT8 + alpha=1 + beta=0 快速路径。
// 使用 int32 累加（LocalTensor 上的 Add<int32_t>）而非 float Cast+Add，
// 对大累加值保持精确的 int32 精度。
// splitK 累加现使用 Vector Add<int32_t>（原地，
// dst==src0）取代标量 GetValue/SetValue 循环。arch35（DAV_3510）支持
// Add<int32_t>。所有 GM<->UB 搬运均使用 DataCopyPad。
// INT8 输出现使用向量化 Cast+clamp（取代标量 clamp 循环）：
//   Cast int32→float（Vector）→ Mins/Maxs clamp（Vector）→ SpltCastFp32ToInt8（Vector）。
// float 中间结果对 clamp 后 [-128,127] 内的值是精确的。
// flag 操作与既有 SpltEpilogueProcessChunk 模式完全一致。
template <typename OutType>
__aicore__ inline void SpltEpilogueInt32AccPath(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<int32_t>& tempGM, GlobalTensor<OutType>& dGM,
    LocalTensor<OutType>& wideCdUB, const SpltCdCtx& cd,
    int64_t base, int32_t row, int32_t colStart,
    int32_t count, int32_t splitK, int64_t totalElem)
{
    DataCopyExtParams copyAcc{1, 0, 0, 0, 0};
    DataCopyPadExtParams<int32_t> padAcc{false, 0, 0, int32_t(0)};
    copyAcc.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));

    // 将 float 缓冲重解释为 int32（sizeof(float) == sizeof(int32_t) == 4）
    LocalTensor<int32_t> accIntUB = accUB.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tempIntUB = tempUB.ReinterpretCast<int32_t>();

    // 1+2. 加载第一个 splitK 段并在 int32 域向量累加其余段
    //      （共享辅助函数，亦被 SpltEpilogueChunkSplitKReduce 的
    //      int32-temp 分支使用）。
    SpltSplitKAccInt32(accIntUB, tempIntUB, tempGM, base, count, splitK, totalElem,
                       copyAcc, padAcc);

    // 3. 输出（X1：ROW 经 SpltCdOffset 单点寻址；COL 走元素块散开）
    if constexpr (std::is_same_v<OutType, int32_t>) {
        // INT32 输出：int32 UB 直接写 GM（无 Cast）
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(row, colStart, cd.ldD, false)], accIntUB, copyAcc);
        } else {
            SpltCdColScatter<OutType>(dGM, accIntUB, wideCdUB, row, colStart, count, cd.ldD);
        }
    } else {
        // INT8 输出：向量化 Cast int32→float→clamp→int8。
        // tempUB（float）复用为 Cast 目的缓冲 —— 上方 splitK 循环完成后
        // 它不再参与累加。
        LocalTensor<int8_t> dUB = dTBuf.Get<int8_t>();
        Cast(tempUB, accIntUB, AscendC::RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Mins(tempUB, tempUB, 127.0f, count);
        Maxs(tempUB, tempUB, -128.0f, count);
        SpltCastFp32ToInt8(dUB, tempUB, halfBuf, count);
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        copyOut.blockLen = static_cast<uint32_t>(count * sizeof(int8_t));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(row, colStart, cd.ldD, false)], dUB, copyOut);
        } else {
            SpltCdColScatter<OutType>(dGM, dUB, wideCdUB, row, colStart, count, cd.ldD);
        }
    }

    // 4. 跨迭代同步（与既有 SpltEpilogueProcessChunk 模式一致）
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

// SpltEpilogueProcessChunk 的子辅助函数。flag 操作与原始内联代码完全一致。

// splitK 加载 + 累加：加载第一段并累加其余 splitK 段。
// int32-temp 路径（INT8 输入）在 int32 域累加，对加载目标缓冲执行
// Add<int32_t>（与 SpltEpilogueInt32AccPath 及 float 分支同模式），
// 加载循环结束后仅做一次 int32->float Cast。MTE2 加载后紧跟逐段
// Cast 并不安全：在 DAV-3510 上 Cast 的完成既不被 MTE2_V 事件握手
// 覆盖，也不被 PipeBarrier<PIPE_V> 覆盖，因此 Cast 可能在下一段
// DataCopyPad 已覆写 tempUB 之后才执行 —— Cast 会读到陈旧的
// （前一段的）内容，reduce 会得到 2*partial0 / 2*partial1。最终 Cast
// 读取的 accIntUB 从不被 MTE2 写入，故不受该竞争影响。循环后的单次
// Cast 对 |sum| < 2^24 也是精确的，且与 golden 参考一致（int32 累加、
// 一次舍入）；逐段 float 累加则会引入每段一次的舍入误差。
template <typename T, typename TempType>
__aicore__ inline void SpltEpilogueChunkSplitKReduce(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB,
    GlobalTensor<TempType>& tempGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem,
    DataCopyExtParams& copyAcc, DataCopyPadExtParams<TempType>& padAcc)
{
    if constexpr (std::is_same_v<TempType, float>) {
        DataCopyPad(accUB, tempGM[base], copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        for (int32_t s = 1; s < splitK; ++s) {
            DataCopyPad(tempUB, tempGM[static_cast<uint64_t>(s) * totalElem + base], copyAcc, padAcc);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Add(accUB, accUB, tempUB, static_cast<int32_t>(count));
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
    } else {
        LocalTensor<int32_t> accIntUB = accUB.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tempIntUB = tempUB.ReinterpretCast<int32_t>();
        SpltSplitKAccInt32(accIntUB, tempIntUB, tempGM, base, count, splitK, totalElem,
                           copyAcc, padAcc);
        // 所有加载消费完毕后做单次 int32->float Cast：accIntUB 仅由
        // V（本 kernel）写入，不存在残留的跨流水竞争。
        Cast(accUB, accIntUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }
}

// 施加 alpha 缩放：D = alpha * acc（标量或本行向量值）。
// 单行 piece 版本：piece 内仅一行，行值一次取出，全部 UB 访问
// 位于缓冲区基址（32B 对齐恒成立）。逐行分段方式以行边界偏移访问
// accUB[offset]/dFp32UB[offset]，FP32 张量需偏移为 8 元素倍数才 32B 对齐，
// 而 FP32 C 描述符仅保证 n%4==0（n%8==4 合法），会触发向量核异常——与
// SpltEpilogueApplyBetaVecRow 的 2 字节 C 张量问题同根因。
template <typename T>
__aicore__ inline void SpltEpilogueApplyAlphaRow(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& accUB, LocalTensor<float>& alphaVecUB,
    int32_t row, int32_t count, int32_t alphaVectorScaling, float alpha)
{
    if (alphaVectorScaling == 1) {
        const float alphaR = SpltGetScalarFromUB(alphaVecUB, row);
        Muls(dFp32UB, accUB, alphaR, count);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(dFp32UB, accUB, alpha, count);
    }
}

// C 行段加载（SpltEpilogueApplyBetaVecRow 与
// SpltEpilogueChunkApplyBetaScalar 的重复开头段提取共享）：ROW 直拷 +
// MTE2_V 握手；COL 经 SpltCdColGather 聚拢。flag 位置与原内联代码
// 逐位一致。
template <typename CType>
__aicore__ inline void SpltEpilogueLoadCRow(
    LocalTensor<CType>& cUB, LocalTensor<CType>& wideCdUB,
    GlobalTensor<CType>& cGM, const SpltCdCtx& cd,
    int32_t row, int32_t colStart, int32_t count,
    DataCopyExtParams& copyC, DataCopyPadExtParams<CType>& padC)
{
    if (!cd.isCol) {
        DataCopyPad(cUB, cGM[SpltCdOffset(row, colStart, cd.ldC, false)], copyC, padC);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        SpltCdColGather<CType>(cUB, wideCdUB, cGM, row, colStart, count, cd.ldC);
    }
}

// 以向量缩放方式施加 beta*C（betaVectorScaling 路径）—— 单行 piece 版本。
// C 的加载/Cast/Muls/Add 全部位于缓冲区基址。逐行分段方式以行边界偏移
// cUB[offset] 作 Cast 源：FP16/BF16 的 2 字节张量需偏移为
// 16 元素倍数才 32B 对齐，而描述符仅保证 n%8==0 —— n%16==8（如
// n=680、n=328）时偏移（如 168 元素 = 336B）非 32B 对齐，Vector Cast
// 触发 ACL_ERROR_RT_VECTOR_CORE_EXCEPTION(507035)。每个 piece 单行、
// 无 chunk 内分段偏移，对任意合法 n/dtype（含 INT8 的 1 字节 C）对齐恒成立。
// GM 侧任意起始地址的行内加载与 fused 路径 SpltFusedEpilogueBetaC 同模式。
// beta*C 的向量缩放路径——单行 piece 版本。CType：C 矩阵实际 dtype
// （读取/Cast 分派全部按 CType：float 直乘 / bf16 向量 Cast / int8 标量
// Cast 循环 / int32 与 half 经 Cast——五分支全既有，零新增 Cast 路径）。
template <typename T, typename CType>
__aicore__ inline void SpltEpilogueApplyBetaVecRow(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    GlobalTensor<CType>& cGM,
    LocalTensor<CType>& wideCdUB, const SpltCdCtx& cd,
    int32_t row, int32_t colStart, int32_t count, float betaR,
    DataCopyExtParams& copyC, DataCopyPadExtParams<CType>& padC)
{
    LocalTensor<CType> cUB = cTBuf.Get<CType>();
    SpltEpilogueLoadCRow<CType>(cUB, wideCdUB, cGM, cd, row, colStart, count, copyC, padC);
    if constexpr (std::is_same_v<CType, float>) {
        Muls(tempUB, cUB, betaR, count);
    } else {
        LocalTensor<float> cFp32UB = cFp32Buf.Get<float>();
        if constexpr (std::is_same_v<CType, __bf16>) {
            SpltCastBf16ToFp32Vec(cFp32UB, cUB, count);
        } else if constexpr (std::is_same_v<CType, int8_t>) {
            SpltScalarCastLoop<float, CType>(cFp32UB, cUB, count);
            PipeBarrier<PIPE_V>();
        } else {
            // half 与 int32_t 共用 CAST_NONE（int32→float 既有先例：
            // SpltFusedEpilogueApplyAlpha 的 AccType=int32 分支）
            Cast(cFp32UB, cUB, AscendC::RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();
        Muls(tempUB, cFp32UB, betaR, count);
    }
    PipeBarrier<PIPE_V>();
    Add(dFp32UB, dFp32UB, tempUB, count);
}

// 以标量 beta 施加 beta*C（非向量缩放路径）——CType 模板化（C 读取与
// Cast 分派同 SpltEpilogueApplyBetaVecRow，五分支全既有零新增）。
template <typename T, typename CType>
__aicore__ inline void SpltEpilogueChunkApplyBetaScalar(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    GlobalTensor<CType>& cGM,
    LocalTensor<CType>& wideCdUB, const SpltCdCtx& cd,
    int32_t row, int32_t colStart, int32_t count,
    DataCopyExtParams& copyC, DataCopyPadExtParams<CType>& padC, float beta)
{
    LocalTensor<CType> cUB = cTBuf.Get<CType>();
    SpltEpilogueLoadCRow<CType>(cUB, wideCdUB, cGM, cd, row, colStart, count, copyC, padC);
    if constexpr (std::is_same_v<CType, float>) {
        Muls(tempUB, cUB, beta, static_cast<int32_t>(count));
    } else {
        LocalTensor<float> cFp32UB = cFp32Buf.Get<float>();
        if constexpr (std::is_same_v<CType, __bf16>) {
            SpltCastBf16ToFp32Vec(cFp32UB, cUB, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<CType, int8_t>) {
            SpltScalarCastLoop<float, CType>(cFp32UB, cUB, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
        } else {
            // half 与 int32_t 共用 CAST_NONE（int32->float 既有先例）
            Cast(cFp32UB, cUB, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(count));
        }
        Muls(tempUB, cFp32UB, beta, static_cast<int32_t>(count));
    }
    Add(dFp32UB, dFp32UB, tempUB, static_cast<int32_t>(count));
}

// ============================================================================
// bias + activation 的 epilogue 辅助函数。
// epilogue 链：SplitK reduce -> alpha -> beta*C -> bias -> activation -> Cast -> 写 D
// bias 和 activation 在 FP32 域（dFp32UB）上操作。
// ============================================================================

// 非融合（piece 式）路径中向 dFp32UB 施加 bias —— 单行 piece 版本。
// 逐行分段 Adds(dFp32UB[offset], ...) 的偏移在 n%8!=0（FP32 C
// 合法输入）时可非 32B 对齐；单行 piece 的基址访问恒对齐（与
// SpltEpilogueApplyAlphaRow 同根因的 32B 对齐约束）。
__aicore__ inline void SpltEpilogueApplyBiasRow(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& biasVecUB,
    int32_t row, int32_t count)
{
    const float biasR = SpltGetScalarFromUB(biasVecUB, row);
    Adds(dFp32UB, dFp32UB, biasR, count);
    PipeBarrier<PIPE_V>();
}

// Cast FP32 -> OutType 并写回 GM（UB -> GM 经 DataCopyPad）。
// X1：ROW 经 SpltCdOffset 单点寻址；COL 走 SpltCdColScatter 元素块散开。
template <typename OutType>
__aicore__ inline void SpltEpilogueChunkStoreOutput(
    LocalTensor<float>& dFp32UB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& halfBuf, GlobalTensor<OutType>& dGM,
    LocalTensor<OutType>& wideCdUB, const SpltCdCtx& cd,
    int32_t row, int32_t colStart, int32_t count,
    DataCopyExtParams& copyAcc, DataCopyExtParams& copyOut)
{
    if constexpr (std::is_same_v<OutType, float>) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(row, colStart, cd.ldD, false)], dFp32UB, copyAcc);
        } else {
            SpltCdColScatter<OutType>(dGM, dFp32UB, wideCdUB, row, colStart, count, cd.ldD);
        }
    } else {
        LocalTensor<OutType> dUB = dTBuf.Get<OutType>();
        if constexpr (std::is_same_v<OutType, int8_t>) {
            Mins(dFp32UB, dFp32UB, 127.0f, static_cast<int32_t>(count));
            Maxs(dFp32UB, dFp32UB, -128.0f, static_cast<int32_t>(count));
            SpltCastFp32ToInt8(dUB, dFp32UB, halfBuf, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<OutType, int32_t>) {
            Cast(dUB, dFp32UB, AscendC::RoundMode::CAST_ROUND, static_cast<int32_t>(count));
        } else if constexpr (std::is_same_v<OutType, __bf16>) {
            SpltCastFp32ToBf16Vec(dUB, dFp32UB, static_cast<int32_t>(count));
        } else {
            Cast(dUB, dFp32UB, AscendC::RoundMode::CAST_ROUND, static_cast<int32_t>(count));
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(row, colStart, cd.ldD, false)], dUB, copyOut);
        } else {
            SpltCdColScatter<OutType>(dGM, dUB, wideCdUB, row, colStart, count, cd.ldD);
        }
    }
}

// INT32 快速路径判定与执行：
// INT32 快速路径：alpha=1+beta=0+无 bias+无 activation 时直接 int32 累加输出
// 有 bias 或 activation 时必须走 FP32 域（bias Adds / GeLU Exp 等在 FP32 域操作）。
// 快速路径仅支持 INT8/INT32 输出（编译期守卫防止 half/bf16 输出实例化
// 其内部 INT8 输出分支）；INT8 A/B × C/D=FP16/BF16（超集组合）走下方
// FP32 域链（int32→float Cast 后既有输出 Cast 写出）。
// 返回 true 表示已按快速路径处理完毕（调用方直接返回）。
template <typename T, typename CType, typename OutType>
__aicore__ inline bool SpltEpilogueTryInt32FastPath(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& wideCdBuf, const SpltCdCtx& cd,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<OutType>& dGM,
    int64_t base, int32_t row, int32_t colStart, int32_t count,
    int32_t splitK, int64_t totalElem,
    int32_t alphaVectorScaling, float alpha, float beta,
    uint64_t biasDevPtr, int32_t activationType)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    if constexpr (!std::is_same_v<TempType, float>
                  && (std::is_same_v<OutType, int8_t> || std::is_same_v<OutType, int32_t>)) {
        if (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
            && biasDevPtr == 0 && activationType == 0) {
            LocalTensor<OutType> wideCdUB = (cd.wideBuf != nullptr)
                ? wideCdBuf.Get<OutType>() : LocalTensor<OutType>{};
            SpltEpilogueInt32AccPath<OutType>(accUB, tempUB, dTBuf, halfBuf, tempGM, dGM,
                                              wideCdUB, cd,
                                              base, row, colStart, count, splitK, totalElem);
            return true;
        }
    }
    return false;
}

// beta*C 施加（向量缩放 / 标量 beta 分派，自
// SpltEpilogueProcessChunk 提取）。wideCdUB 仅 COL 路径消费（ROW 时
// wideCdBuf 未 InitBuffer，构造空 tensor 不触访存）。
template <typename T, typename CType>
__aicore__ inline void SpltEpilogueApplyBeta(
    LocalTensor<float>& dFp32UB, LocalTensor<float>& tempUB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& wideCdBuf, const SpltCdCtx& cd,
    GlobalTensor<CType>& cGM, LocalTensor<float>& betaVecUB,
    int32_t row, int32_t colStart, int32_t count,
    DataCopyExtParams& copyC, DataCopyPadExtParams<CType>& padC,
    float beta, int32_t betaVectorScaling)
{
    LocalTensor<CType> wideCdUB = (cd.wideBuf != nullptr)
        ? wideCdBuf.Get<CType>() : LocalTensor<CType>{};
    if (betaVectorScaling == 1) {
        const float betaR = SpltGetScalarFromUB(betaVecUB, row);
        SpltEpilogueApplyBetaVecRow<T, CType>(dFp32UB, tempUB, cTBuf, cFp32Buf,
                                               cGM, wideCdUB, cd,
                                               row, colStart, count, betaR, copyC, padC);
    } else {
        SpltEpilogueChunkApplyBetaScalar<T, CType>(dFp32UB, tempUB, cTBuf, cFp32Buf,
                                                    cGM, wideCdUB, cd,
                                                    row, colStart, count, copyC, padC, beta);
    }
}

// 处理 epilogue 循环的一个 piece（splitK reduce + alpha/beta + 输出）。
// piece = 单行内长度 ≤ CHUNK 的片段（由 SpltEpilogueBatchLoop 的
// 行内 piece 迭代产生），向量缩放/bias 的逐行操作退化为单段基址访问。
// 所有 SetFlag/WaitFlag 保持不变。
// 模板化为 (T, OutType)。AccType 恒为 float（非快速路径下 INT8 的
// int32 temp 在加载时 Cast 为 float）。INT8 输出在 Cast 前用
// Mins/Maxs clamp 到 [-128,127]（CAST_SATURATE 不可用）。
// halfBuf 参数：float→int8_t 需要两步 Cast。
// alpha=1+beta=0 时经 int32 快速路径分发（不经过 float，
// SpltEpilogueTryInt32FastPath）；beta 分派见 SpltEpilogueApplyBeta。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltEpilogueProcessChunk(
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB, LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& wideCdBuf, const SpltCdCtx& cd,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    int64_t base, int32_t row, int32_t colStart, int32_t count,
    int32_t splitK, float alpha, float beta,
    int64_t totalElem,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    LocalTensor<float>& biasVecUB,
    LocalTensor<uint8_t>& maskUB,
    uint64_t biasDevPtr, int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    // INT32 快速路径（命中则处理完毕直接返回）。
    if (SpltEpilogueTryInt32FastPath<T, CType, OutType>(
            accUB, tempUB, dTBuf, halfBuf, wideCdBuf, cd, tempGM, dGM,
            base, row, colStart, count, splitK, totalElem,
            alphaVectorScaling, alpha, beta, biasDevPtr, activationType)) {
        return;
    }
    DataCopyExtParams copyAcc{1, 0, 0, 0, 0};
    DataCopyPadExtParams<TempType> padAcc{false, 0, 0, TempType(0)};
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    DataCopyPadExtParams<OutType> padOut{false, 0, 0, OutType(0)};
    DataCopyExtParams copyC{1, 0, 0, 0, 0};
    DataCopyPadExtParams<CType> padC{false, 0, 0, CType(0)};
    copyAcc.blockLen = static_cast<uint32_t>(count * sizeof(TempType));
    copyOut.blockLen = static_cast<uint32_t>(count * sizeof(OutType));
    copyC.blockLen = static_cast<uint32_t>(count * sizeof(CType));

    SpltEpilogueChunkSplitKReduce<T, TempType>(accUB, tempUB, tempGM,
                                                base, count, splitK, totalElem, copyAcc, padAcc);
    SpltEpilogueApplyAlphaRow<T>(dFp32UB, accUB, alphaVecUB,
                                  row, count, alphaVectorScaling, alpha);
    if (betaVectorScaling == 1 || beta != 0.0f) {
        SpltEpilogueApplyBeta<T, CType>(dFp32UB, tempUB, cTBuf, cFp32Buf, wideCdBuf, cd,
                                        cGM, betaVecUB, row, colStart, count, copyC, padC,
                                        beta, betaVectorScaling);
    }
    // ⑤ bias → 累加到 dFp32UB
    if (biasDevPtr != 0) {
        SpltEpilogueApplyBiasRow(dFp32UB, biasVecUB, row, count);
    }
    // ⑥ activation
    if (activationType == 1) {
        // thr 暂存复用 tempUB（此时 beta 中间量已消费完毕）
        SpltApplyReLU(dFp32UB, tempUB, maskUB, static_cast<int32_t>(count),
                      reluThreshold, reluUpperBound);
    } else if (activationType == 2) {
        // 非融合路径：piece 级批量执行 GeLU，geluTemp 复用 tempUB
        SpltApplyGeLU(dFp32UB, tempUB, static_cast<int32_t>(count), geluScaling);
    }
    {
        LocalTensor<OutType> wideCdUB = (cd.wideBuf != nullptr)
            ? cd.wideBuf->Get<OutType>() : LocalTensor<OutType>{};
        SpltEpilogueChunkStoreOutput<OutType>(dFp32UB, dTBuf, halfBuf, dGM,
                                               wideCdUB, cd,
                                               row, colStart, count, copyAcc, copyOut);
    }
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

// SpltEpilogueImpl 的 TBuf 初始化（按 CHUNK 分配）。
// cTBuf 按 max(sizeof(T), sizeof(CType))（合法域恒 CType >= T，防御性取
// max）；cFp32Buf 条件键于 CType（C=INT32/INT8/FP16/BF16 均需 FP32 中间缓冲）。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltEpilogueInitBuffers(
    TPipe& pipe,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& wideCdBuf, bool cOrderIsCol,
    float beta, int32_t betaVectorScaling)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    constexpr int32_t CHUNK = 256;
    pipe.InitBuffer(accBuf, CHUNK * sizeof(float));
    pipe.InitBuffer(tempBuf, CHUNK * sizeof(float));
    pipe.InitBuffer(dFp32Buf, CHUNK * sizeof(float));
    if constexpr (!std::is_same_v<TempType, float>) {
        pipe.InitBuffer(dTBuf, CHUNK * sizeof(float));
    }
    if constexpr (std::is_same_v<OutType, int8_t>) {
        pipe.InitBuffer(halfBuf, CHUNK * sizeof(half));
    }
    // X1：COL 中转缓冲（每元素一个 32B 槽位，容量 = 单 piece 最大段长）。
    if (cOrderIsCol) {
        pipe.InitBuffer(wideCdBuf, static_cast<uint32_t>(CHUNK) * SPLT_CD_COL_BLOCK_BYTES);
    }
    // 移除 BF16 的 halfBuf 分配：SpltCastFp32ToBf16Vec 不使用
    // halfBuf（Vector Cast<bfloat16_t,float> 无需中间缓冲）。
    if (betaVectorScaling == 1 || beta != 0.0f) {
        const uint32_t cElemSize = (sizeof(T) > sizeof(CType)) ? sizeof(T) : sizeof(CType);
        pipe.InitBuffer(cTBuf, CHUNK * cElemSize);
        if constexpr (!std::is_same_v<CType, float>) {
            pipe.InitBuffer(cFp32Buf, CHUNK * sizeof(float));
        }
    }
    if constexpr (!std::is_same_v<OutType, float> && std::is_same_v<TempType, float>) {
        pipe.InitBuffer(dTBuf, CHUNK * sizeof(OutType));
    }
}

// 为 SpltEpilogueBatchLoop 设置逐 batch 的 GM 指针。
// 将 temp/c/d/bias GM 按 batch stride 偏移；biasStride != 0 时重载 bias。
// CType：C 矩阵实际 dtype（batchStrideC 以 CType 元素为单位——类型化
// 指针偏移自动正确）。bias 恒键于 T。
// C/D/bias 的 batch 基址段经 SpltOffsetBatchCdBiasGM
// 共享（融合侧 SpltFusedEpilogueOffsetBatchGM 同款）。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltEpilogueOffsetBatchGM(
    const AclsparseltTilingData& td,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    int32_t b, int32_t m, int64_t totalElem)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;
    using BiasType = SpltBiasDType<T>;
    const int64_t tempBatchElems = static_cast<int64_t>(td.splitK) * totalElem;
    tempGM.SetGlobalBuffer(reinterpret_cast<__gm__ TempType*>(tempGm)
                           + static_cast<int64_t>(b) * tempBatchElems,
                           static_cast<uint64_t>(tempBatchElems));
    // [附注1] C/D 长度按物理 span（ROW: (m-1)*ld+n / COL: (n-1)*ld+m，经
    // SpltCdSpanElems）——ld>min / COL 时逻辑 m*n 小于实际物理跨度；temp
    // 恒为逻辑 (m,n) 行主序（m*n 口径不变）。C/D 同 order（描述符层
    // ValidateCDConsistency 保证）。
    SpltOffsetBatchCdBiasGM<CType, OutType, BiasType>(td, cGM, dGM, biasVecGM,
                                                      cGm, dGm, b, m);
    // 非融合路径 bias 恒全量重载（chunk 降级仅存在于融合路径）。
    if (td.biasDevPtr != 0 && td.biasStride != 0) {
        SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
    }
}

// SpltEpilogueImpl 的 batch 循环：遍历 batch，设置逐 batch GM 指针、
// 按需重载 bias，并执行 chunk 处理循环。
// flag 操作与循环顺序完全保持不变。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltEpilogueBatchLoop(
    const AclsparseltTilingData& td,
    GlobalTensor<typename SpltL0CTypeTrait<T>::type>& tempGM,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& accUB, LocalTensor<float>& tempUB, LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& cTBuf, TBuf<TPosition::VECCALC>& cFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& wideCdBuf, const SpltCdCtx& cd,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    LocalTensor<uint8_t>& maskUB,
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling, float beta)
{
    const int32_t n = td.n;
    const int32_t splitK = td.splitK;
    const float alpha = td.alpha;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int32_t activationType = td.activationType;
    const float reluThreshold = td.reluThreshold;
    const float reluUpperBound = td.reluUpperBound;
    const float geluScaling = td.geluScaling;
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;
    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const int64_t totalElem = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    constexpr int32_t CHUNK = 256;

    for (int32_t b = 0; b < numBatches; ++b) {
        SpltEpilogueOffsetBatchGM<T, CType, OutType>(td, tempGM, cGM, dGM, biasVecGM,
                                              biasVecUB, biasRawBuf,
                                              tempGm, cGm, dGm, b, m, totalElem);

        // 行内 piece 迭代（每 piece 位于单一矩阵行内，长度 ≤ CHUNK）：
        // 向量缩放/bias 的逐行操作不需要 chunk 内分段偏移，全部 UB 访问
        // 位于缓冲区基址，32B 对齐对任意合法 n/dtype 恒成立——若以行边界
        // 偏移访问 cUB[offset]（FP16/BF16 需 16 元素倍数）/
        // dFp32UB[offset]（FP32 需 8 元素倍数），则 n%16==8 / n%8==4 的
        // 合法 shape 下偏移非 32B 对齐，Vector 指令触发
        // ACL_ERROR_RT_VECTOR_CORE_EXCEPTION(507035)。
        // 逐元素数学与逐行分段实现逐位一致（piece 化仅重新分组指令，
        // 不改变逐元素的数值序）；n%256==0 时 piece 边界与 256 元素
        // chunk 边界重合，调度不变。
        const int32_t piecesPerRow = (n + CHUNK - 1) / CHUNK;
        const int64_t totalPieces = static_cast<int64_t>(m) * piecesPerRow;
        for (int64_t piece = blockId; piece < totalPieces; piece += blockNum) {
            const int32_t row = static_cast<int32_t>(piece / piecesPerRow);
            const int32_t j = static_cast<int32_t>(piece % piecesPerRow);
            // base 仅承载 temp（行主序逻辑 (m,n) 布局，与 C/D 的 ld/order
            // 正交）；C/D 的寻址经 SpltCdOffset(row, colStart, ...) 单点
            // 派生（X1）。
            const int64_t base = static_cast<int64_t>(row) * n + static_cast<int64_t>(j) * CHUNK;
            const int32_t colStart = j * CHUNK;
            const int32_t count = (j + 1) * CHUNK <= n ? CHUNK : (n - j * CHUNK);
            SpltEpilogueProcessChunk<T, CType, OutType>(accUB, tempUB, dFp32UB, cTBuf, cFp32Buf, dTBuf, halfBuf,
                                        wideCdBuf, cd,
                                        tempGM, cGM, dGM, base, row, colStart, count, splitK, alpha, beta,
                                        totalElem, alphaVectorScaling, betaVectorScaling,
                                        alphaVecUB, betaVecUB,
                                        biasVecUB, maskUB, biasDevPtr, activationType,
                                        reluThreshold, reluUpperBound, geluScaling);
        }
    }
}

// ============================================================================
// SpltEpilogueImpl 模板化为 (T, CType, OutType)。AccType 恒为 float
// （INT8 的 int32 temp 在加载时 Cast 为 float）。CType 为 C 矩阵实际
// dtype（INT8 A/B 超集行下可与 T 不同）；非 INT8 路径 CType == OutType
// == T。SpltCdCtx 构造（splt_make_cd_ctx）与 ReLU 掩码缓冲
// （splt_init_relu_mask）为 matmul_kernel_common.h 的共享 helper
// （融合 Impl 共享）。
template <typename T, typename CType = T, typename OutType = T>
__aicore__ inline void SpltEpilogueImpl(GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    using TempType = typename SpltL0CTypeTrait<T>::type;  // float 或 int32_t
    using BiasType = SpltBiasDType<T>;  // bias dtype：FP32/FP16/BF16 时为 T，INT8 时为 float
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const float beta = td.beta;
    const int32_t alphaVectorScaling = td.alphaVectorScaling;
    const int32_t betaVectorScaling = td.betaVectorScaling;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int64_t biasStride = td.biasStride;

    GlobalTensor<TempType> tempGM;
    GlobalTensor<CType> cGM;
    GlobalTensor<OutType> dGM;

    GlobalTensor<float> alphaVecGM;
    GlobalTensor<float> betaVecGM;
    SpltInitScalingGM(alphaVecGM, betaVecGM, td.alphaDevPtr, td.betaDevPtr,
                       m, alphaVectorScaling, betaVectorScaling);

    GlobalTensor<BiasType> biasVecGM;
    TBuf<TPosition::VECCALC> biasVecBuf;
    TBuf<TPosition::VECCALC> biasRawBuf;
    LocalTensor<float> biasVecUB;
    TPipe pipe;
    if (biasDevPtr != 0) {
        SpltInitBiasBuffers<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                       pipe, biasDevPtr, biasStride, m);
    }

    TBuf<TPosition::VECCALC> accBuf, tempBuf, cTBuf, cFp32Buf, dFp32Buf, dTBuf, halfBuf;
    TBuf<TPosition::VECCALC> alphaVecBuf, betaVecBuf;
    TBuf<TPosition::VECCALC> wideCdBuf;
    SpltCdCtx cd = splt_make_cd_ctx(wideCdBuf, td.ldC, td.ldD, td.cOrderIsCol != 0);
    SpltEpilogueInitBuffers<T, CType, OutType>(pipe, accBuf, tempBuf, dFp32Buf, dTBuf,
                                          cTBuf, cFp32Buf, halfBuf, wideCdBuf, cd.isCol,
                                          beta, betaVectorScaling);
    LocalTensor<float> alphaVecUB;
    LocalTensor<float> betaVecUB;
    // two-stage 恒全量加载（scalingChunkMode=0）：splitK>1 的
    // biasChunkMode 恒为 0（host 侧 compute_bias_chunk_mode 条件限定
    // splitK==1），超限域由 host 侧 validate_scaling_ub_budget 拒绝。
    SpltLoadScalingUB(alphaVecBuf, betaVecBuf, alphaVecUB, betaVecUB,
                       alphaVecGM, betaVecGM, pipe, m,
                       alphaVectorScaling, betaVectorScaling, 0, 0);
    // ReLU b8 位掩码缓冲：piece <= CHUNK 元素 -> <= 32 字节（64B 对齐分配）
    TBuf<TPosition::VECCALC> reluMaskBuf;
    LocalTensor<uint8_t> reluMaskUB = splt_init_relu_mask(pipe, reluMaskBuf);

    LocalTensor<float> accUB = accBuf.Get<float>();
    LocalTensor<float> tempUB = tempBuf.Get<float>();
    LocalTensor<float> dFp32UB = dFp32Buf.Get<float>();

    SpltEpilogueBatchLoop<T, CType, OutType>(td, tempGM, cGM, dGM, biasVecGM, biasVecUB, biasRawBuf,
        accUB, tempUB, dFp32UB, cTBuf, cFp32Buf, dTBuf, halfBuf, wideCdBuf, cd,
        alphaVecUB, betaVecUB,
        reluMaskUB, tempGm, cGm, dGm, m, alphaVectorScaling, betaVectorScaling, beta);
}

// --- epilogue ---
extern "C" __global__ __aicore__ void splt_epilogue_kernel_fp16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<__fp16>(tempGm, cGm, dGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_epilogue_kernel_fp32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<float>(tempGm, cGm, dGm, tilingGm);
}
// BF16 epilogue。
// BF16 Cast（FP32<->BF16）使用 Vector Cast（CANN 9.1.0
// 在 cast_round_all 中支持 Cast<bfloat16_t,float>，在 cast_none 中
// 支持 Cast<float,bfloat16_t>）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_bf16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<__bf16>(tempGm, cGm, dGm, tilingGm);
}
// INT8 epilogue —— INT8 输出（饱和 Cast）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int8_t>(tempGm, cGm, dGm, tilingGm);
}
// INT8 epilogue —— INT32 输出（int32 直接写）。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_i32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int8_t, int32_t>(tempGm, cGm, dGm, tilingGm);
}
// INT8 A/B × C/D 超集组合（CType 贯通）：C=D=INT32 / FP16 / BF16。
// 数值路径：int8×int8 -> int32 L0C 累加 -> FP32 epilogue -> 按 OutType 写出。
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_ci32(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, int32_t, int32_t>(tempGm, cGm, dGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_cf16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, __fp16, __fp16>(tempGm, cGm, dGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_epilogue_kernel_int8_cbf16(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpltEpilogueImpl<int8_t, __bf16, __bf16>(tempGm, cGm, dGm, tilingGm);
}

// outDataType —— INT8 路径选择 INT8（饱和 Cast）或 INT32（直接写）。
extern "C" void splt_epilogue_kernel_launch(
    GM_ADDR tempGm, GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t cDataType, int32_t outDataType,
    uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_epilogue_kernel_fp32<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_epilogue_kernel_fp16<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_epilogue_kernel_bf16<<<blockDim, nullptr, stream>>>(
                tempGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            // C/D 超集分派：C=D=INT32 / FP16 / BF16（CType 贯通）；
            // C=INT8 时按 outDataType 分 INT8/INT32（既有）。
            if (cDataType == SPLT_DTYPE_INT32) {
                splt_epilogue_kernel_int8_ci32<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            } else if (cDataType == SPLT_DTYPE_FP16) {
                splt_epilogue_kernel_int8_cf16<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            } else if (cDataType == SPLT_DTYPE_BF16) {
                splt_epilogue_kernel_int8_cbf16<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            } else if (outDataType == SPLT_DTYPE_INT32) {
                splt_epilogue_kernel_int8_i32<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            } else {
                // 不可达兜底（防御保留）：host 侧 launch 前置校验
                // （validate_launch_dtypes 配对白名单）已保证到达本分支的
                // INT8 组合为 (C=INT8, D=INT8)——其余 C/D 组合在 launch 前
                // 即被拒绝（NOT_SUPPORTED）。
                splt_epilogue_kernel_int8<<<blockDim, nullptr, stream>>>(
                    tempGm, cGm, dGm, tilingGm);
            }
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in epilogue launcher\n"); return;
    }
}
