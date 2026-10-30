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
 * \file matmul_kernel_common.h
 * \brief sparseLt matmul kernel 族的公共设备侧基础设施（超大
 *        源文件整改：原 matmul_kernel.cpp nbnc 2147 > 2000，按功能族拆为
 *        matmul_kernel.cpp（非融合 cube）/ matmul_epilogue_kernel.cpp（非融
 *        合 epilogue）/ matmul_fused_kernel.cpp（融合 cube+epilogue）三个
 *        ASC 编译单元，本头承载跨 TU 共享的 trait / Cast / C-D 布局寻址 /
 *        K 循环 / bias 与缩放向量初始化 / ReLU-GeLU 等辅助函数。全部为
 *        __aicore__ inline 定义（含模板），逐 TU 内联展开，无 ODR 风险。
 */

#ifndef SPLT_MATMUL_KERNEL_COMMON_H
#define SPLT_MATMUL_KERNEL_COMMON_H

#include <cstdint>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "shared/aclsparselt_internal.h"

#ifdef __CCE_AICORE__
static constexpr const char* kSparseLtLogTag = "aclsparseLt";
#endif

using namespace AscendC;
// tensor_api（Te 命名空间）的符号统一以显式 Te:: 前缀引用，
// 以避免 AscendC（kernel_operator.h）与 AscendC::Te（tensor_api）之间
// LocalTensor/GlobalTensor 的二义性。禁止添加 "using namespace AscendC::Te"。

// ============================================================================
// L0C 累加类型映射。
// FP32/FP16/BF16 以 float 累加（L0C=float）；INT8 以 int32_t 累加。
// 用于在构造处（cube + fused）推导 L0C 张量类型。
// ============================================================================
template <typename T> struct SpltL0CTypeTrait { using type = float; };  // FP32/FP16/BF16
template <> struct SpltL0CTypeTrait<int8_t> { using type = int32_t; };  // INT8

// ============================================================================
// bias dtype mapping (per the cuSPARSELt spec, keyed on (T, CType)):
// "bias 向量的数据类型与矩阵 C 相同，唯一例外是 INT8 输入、INT8/INT32 输出、
//  INT32 计算的场景——此时 bias 的数据类型为 FP32。"
//   FP32 路径：bias = FP32（= C dtype）→ 直接加载，无需 Cast
//   FP16 路径：bias = FP16（= C dtype）→ 加载 FP16，Cast 到 FP32
//   BF16 路径：bias = BF16（= C dtype）→ 加载 BF16，Cast 到 FP32
//   INT8 × C/D ∈ {INT8, INT32}（32I 计算域）：bias = FP32 → 直接加载
//   INT8 × C/D=FP16/BF16：bias = FP16/BF16（= C dtype）→ 加载后 Cast 到
//   FP32（INT8 输入不改变 bias=C dtype 规则，例外仅限 32I 计算域）
// 五组合正确 bias dtype 表：int8×(C8,D8)→FP32；int8×(C8,D32)→FP32；
// int8×(C32,D32)→FP32；int8×(C16,D16)→FP16；int8×(CBF16,DBF16)→BF16。
// 非 INT8 路 CType == T，主模板退化为原行为（零回归锚点）。
// ============================================================================
template <typename T, typename CType>
struct SpltBiasDTypeTrait { using type = CType; };  // primary template: bias = C dtype
template <> struct SpltBiasDTypeTrait<int8_t, int8_t> { using type = float; };   // 32I-domain exception
template <> struct SpltBiasDTypeTrait<int8_t, int32_t> { using type = float; };  // 32I-domain exception
template <typename T, typename CType>
using SpltBiasDType = typename SpltBiasDTypeTrait<T, CType>::type;

// 逐元素标量 Cast 循环（GetValue/SetValue）。对支持 static_cast 的
// 类型（FP16、INT8 等）使用 static_cast。
template <typename DstT, typename SrcT>
__aicore__ inline void SpltScalarCastLoop(LocalTensor<DstT>& dst, LocalTensor<SrcT> src,
                                           int32_t count)
{
    for (int32_t i = 0; i < count; ++i) {
        dst.SetValue(i, static_cast<DstT>(src.GetValue(i)));
    }
}

// ----------------------------------------------------------------------------
// float→int8_t Cast 辅助函数。
// DAV-3510 的 Cast API 不支持 float→int8_t 直接转换
// （CastImpl 的 cast_round_all / cast_none 列表中没有 Tuple<int8_t, float>）。
// release 模式下 ASCENDC_ASSERT 为空操作，Cast 会静默不执行——
// 目的缓冲区未被写入，产生垃圾数据。
// 受支持的路径：float→half（CAST_ROUND，Tuple<half,float> 在 cast_round_all 中）
// → int8_t（CAST_ROUND，Tuple<int8_t,half> 在 cast_round_all 中）。两步均为
// 硬件 Vector Cast。half 中间结果可放入一个很小的 TBuf
// （count * sizeof(half) 字节）。数值已由调用方预先 clamp 到 [-128,127]
// （Mins/Maxs），且 half 能精确表示该范围内的所有整数。
// ----------------------------------------------------------------------------
__aicore__ inline void SpltCastFp32ToInt8(
    LocalTensor<int8_t>& dst, const LocalTensor<float>& src,
    TBuf<TPosition::VECCALC>& halfBuf, int32_t count)
{
    LocalTensor<half> halfUB = halfBuf.Get<half>();
    Cast(halfUB, src, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
    Cast(dst, halfUB, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
}

// ----------------------------------------------------------------------------
// 向量化 FP32→BF16 Cast。
// DAV-3510 CANN 9.1.0 支持 Vector Cast<bfloat16_t, float, CAST_ROUND>
// （Tuple<bfloat16_t, float> 在 cast_round_all 中）。单次 Vector Cast
// 调用即可达到与 FP16 路径相当的性能特征。
// ----------------------------------------------------------------------------
__aicore__ inline void SpltCastFp32ToBf16Vec(
    LocalTensor<bfloat16_t>& dst, const LocalTensor<float>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
}

// 向量化 BF16→FP32 Cast（SpltCastFp32ToBf16Vec 的逆操作）。
// 支持 Cast<float, bfloat16_t, CAST_NONE>（Tuple<float, bfloat16_t>
// 在 cast_none 中）。BF16→FP32 是加宽 Cast（无精度损失 → CAST_NONE）。
__aicore__ inline void SpltCastBf16ToFp32Vec(
    LocalTensor<float>& dst, const LocalTensor<bfloat16_t>& src, int32_t count)
{
    Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
}

// ============================================================================
// X1：C/D 的 ld + order 布局语义接入（单点寻址 + 段搬运 helper）。
//
// SpltCdOffset —— 单点寻址函数：逻辑元素 (rowIdx, colIdx)
// 的物理偏移。ROW：rowIdx*ld + colIdx（行步长 ld，ld == n 时与旧
// row*n + col 寻址逐位等价）；COL：colIdx*ld + rowIdx（列主序，元素
// (i,j) 位于 i + j*ld）。7 处 C/D 寻址点统一经本函数取块起点。
// ============================================================================
__aicore__ inline int64_t SpltCdOffset(int64_t rowIdx, int64_t colIdx,
                                       int32_t ld, bool isCol)
{
    return isCol ? (colIdx * static_cast<int64_t>(ld) + rowIdx)
                 : (rowIdx * static_cast<int64_t>(ld) + colIdx);
}

// [附注1] C/D 的物理跨度（元素数）：cGM/dGM 的 SetGlobalBuffer 长度按
// 物理布局口径而非逻辑 m*n——ROW: (m-1)*ld + n；COL: (n-1)*ld + m。ld > min
// / COL 布局时物理跨度大于 m*n（padding 域属同一物理缓冲，m*n 口径偏小）。
// ld ≥ min 由描述符层（DenseDescriptorInit 的 IsLdValid）保证，max 防御
// 病态输入时回缩小于 m*n。紧凑域（ROW ld==n / COL ld==m）下恒等于 m*n，
// 与既有值逐位一致（零回归锚点）。
__aicore__ inline uint64_t SpltCdSpanElems(int32_t m, int32_t n, int32_t ld, bool isCol)
{
    const int64_t span = isCol
        ? (static_cast<int64_t>(n) - 1) * static_cast<int64_t>(ld) + m
        : (static_cast<int64_t>(m) - 1) * static_cast<int64_t>(ld) + n;
    const int64_t mn = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    return static_cast<uint64_t>(span > mn ? span : mn);
}

// C/D 布局上下文：从 td 派生一次，经 epilogue 链传递（替代散装的
// ldC/ldD/isCol 标量参数）。wideBuf 为 COL 中转缓冲（每元素一个 32B
// 槽位，容量 = 行内最大段长），非 COL 路径为 nullptr（零开销）。
struct SpltCdCtx {
    int32_t ldC = 0;
    int32_t ldD = 0;
    bool isCol = false;
    TBuf<TPosition::VECCALC>* wideBuf = nullptr;
};

// SpltCdCtx 构造（非融合/融合两个 epilogue Impl
// 的同型内联段提取共享）：ld/order 来自 td，wideBuf 仅 COL 路径指向中转
// 缓冲（ROW 时为 nullptr，零开销）。
__aicore__ inline SpltCdCtx splt_make_cd_ctx(
    TBuf<TPosition::VECCALC>& wideCdBuf, int32_t ldC, int32_t ldD, bool isCol)
{
    SpltCdCtx cd;
    cd.ldC = ldC;
    cd.ldD = ldD;
    cd.isCol = isCol;
    cd.wideBuf = (isCol) ? &wideCdBuf : nullptr;
    return cd;
}

// ReLU b8 位掩码缓冲初始化（非融合 piece 级
// <=CHUNK 元素 / 融合行级 <=maxNAlign 元素，64B 分配覆盖两种粒度）。
__aicore__ inline LocalTensor<uint8_t> splt_init_relu_mask(
    TPipe& pipe, TBuf<TPosition::VECCALC>& reluMaskBuf)
{
    pipe.InitBuffer(reluMaskBuf, 64);
    return reluMaskBuf.Get<uint8_t>();
}

// —— COL 段搬运（元素块 2D + 标量聚拢/散开）——
// 模型：COL 布局下逻辑行 rowIdx 的 [colStart, colStart+count) 段在物理上
// 是 count 个间隔 ld 的单元素（(rowIdx, colStart+c) 位于 rowIdx +
// (colStart+c)*ld）。GM 侧经 2D DataCopyPad 的元素块形态访问（块 = 1 元素，
// 块间 stride = (ld-1)*elemSize，GM 侧 stride 单位为字节且无 32B 对齐
// 要求——同仓 spsm SpsmTransposeAIV 先例）；UB 侧 2D 块自动按 32B 槽位
// 排放（prune SpltPruneLoadRowBlock 的 dstStride=0 布局语义），与向量
// 计算所需的紧凑布局不一致——经标量 GetValue/SetValue 聚拢/散开衔接
// （UB LocalTensor 标量访问，prune 大量先例；黑名单仅限 GM 侧）。
// 同步：stride>0 的 2D 搬运 blockCount 受 ~29-32 行隐式上限约束
// （api-datacopy.md，超出块静默丢弃），保守取 16（远低于实测边界，
// 也覆盖 fused 批量分支的 32 行分批上限）。
// 性能注：COL 为非主流布局（既有用例 C/D 恒 ROW），本实现正确性优先，
// 每段额外 ~2*count 次 UB 标量访问；后续可平滑替换为 TransDataTo5HD
// 转置方案（接口不变）。
constexpr int32_t SPLT_CD_COL_BLOCK = 16;
// COL 中转缓冲的槽位粒度：UB 侧 2D 块按 32B 边界排放（每元素占一个
// 32B 槽），缓冲大小 = 段长（元素数）× 32B。
constexpr int32_t SPLT_CD_COL_BLOCK_BYTES = 32;

// COL C 读：逻辑行 rowIdx 的 count 元素段 GM→UB 紧凑（聚拢）。
// 出口同步：PipeBarrier<PIPE_V>（标量写后供向量指令消费，prune 先例）。
template <typename T>
__aicore__ inline void SpltCdColGather(LocalTensor<T>& dstUB, LocalTensor<T>& wideUB,
                                       GlobalTensor<T>& cGM, int64_t rowIdx, int32_t colStart,
                                       int32_t count, int32_t ld)
{
    constexpr int32_t kEltBytes = sizeof(T);
    constexpr int32_t kSlotsPerBlock = 32 / kEltBytes;  // 32B 槽位内的元素粒度
    DataCopyPadExtParams<T> padC{false, 0, 0, T(0)};
    int32_t done = 0;
    while (done < count) {
        const int32_t cur = (count - done > SPLT_CD_COL_BLOCK) ? SPLT_CD_COL_BLOCK : (count - done);
        DataCopyExtParams copyC{static_cast<uint16_t>(cur), static_cast<uint32_t>(kEltBytes),
                                static_cast<int64_t>(ld - 1) * kEltBytes, 0, 0};
        DataCopyPad(wideUB[static_cast<int64_t>(done) * kSlotsPerBlock],
                    cGM[rowIdx + static_cast<int64_t>(colStart + done) * ld], copyC, padC);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        done += cur;
    }
    // 标量聚拢：32B 槽位布局 → 紧凑段
    for (int32_t i = 0; i < count; ++i) {
        dstUB.SetValue(i, wideUB.GetValue(static_cast<int32_t>(i) * kSlotsPerBlock));
    }
    PipeBarrier<PIPE_V>();
}

// COL D 写：逻辑行 rowIdx 的 count 元素段 UB 紧凑→GM（散开）。
// 入口同步：PipeBarrier<PIPE_V>（向量产物供标量读取）；出口同步：
// MTE3_MTE2（跨迭代 WAR，防下一段 Gather 覆写 wideUB）。
template <typename T>
__aicore__ inline void SpltCdColScatter(GlobalTensor<T>& dGM, const LocalTensor<T>& srcUB,
                                        LocalTensor<T>& wideUB, int64_t rowIdx, int32_t colStart,
                                        int32_t count, int32_t ld)
{
    constexpr int32_t kEltBytes = sizeof(T);
    constexpr int32_t kSlotsPerBlock = 32 / kEltBytes;
    PipeBarrier<PIPE_V>();  // 向量产物 → 标量读取
    // 标量散开：紧凑段 → 32B 槽位布局
    for (int32_t i = 0; i < count; ++i) {
        wideUB.SetValue(static_cast<int32_t>(i) * kSlotsPerBlock, srcUB.GetValue(i));
    }
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    int32_t done = 0;
    while (done < count) {
        const int32_t cur = (count - done > SPLT_CD_COL_BLOCK) ? SPLT_CD_COL_BLOCK : (count - done);
        DataCopyExtParams copyD{static_cast<uint16_t>(cur), static_cast<uint32_t>(kEltBytes), 0,
                                static_cast<int64_t>(ld - 1) * kEltBytes, 0};
        DataCopyPad(dGM[rowIdx + static_cast<int64_t>(colStart + done) * ld],
                    wideUB[static_cast<int64_t>(done) * kSlotsPerBlock], copyD);
        done += cur;
    }
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

// ============================================================================
// cube matmul kernel 的共享辅助函数。
// 关键约束：SetFlag/WaitFlag 的位置与原始内联代码完全一致——
// 禁止重排或合并任何 flag 操作。
// ============================================================================
struct L1BufferConfig {
    int64_t aL1Off[2];
    int64_t bL1Off[2];
};

// L1 双缓冲初始化：buffer 大小按 kL1Size（而非
// baseK）计算，使外层 kL1 循环每次向 L1 加载更大的 K 块。预置
// MTE1_MTE2(0/1) 使首个 WaitFlag 可通过；M_MTE1(0/1) 用于 L0 乒乓；
// FIX_M(0) 用于跨 tile 的 L0C 复用握手。
template <typename T>
__aicore__ inline L1BufferConfig InitL1DoubleBuffer(int32_t baseM, int32_t baseN, int32_t kL1Size)
{
    const int64_t aL1Bytes = static_cast<int64_t>(baseM) * kL1Size * sizeof(T);
    const int64_t bL1Bytes = static_cast<int64_t>(kL1Size) * baseN * sizeof(T);
    L1BufferConfig cfg;
    cfg.aL1Off[0] = 0;
    cfg.aL1Off[1] = aL1Bytes + bL1Bytes;
    cfg.bL1Off[0] = aL1Bytes;
    cfg.bL1Off[1] = 2 * aL1Bytes + bL1Bytes;
    SetFlag<HardEvent::MTE1_MTE2>(0);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_MTE1>(1);
    SetFlag<HardEvent::FIX_M>(0);
    return cfg;
}

// kernel 结束时排空 cube 侧全部 flag：
// - L1 乒乓：MTE1_MTE2(0/1) + M_MTE1(0/1) —— L1 双缓冲同步
// - L0C 复用：FIX_M(0) —— 跨 tile 的 L0C 复用握手（非双缓冲 flag）
__aicore__ inline void DrainCubeFlags()
{
    WaitFlag<HardEvent::MTE1_MTE2>(0);
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::M_MTE1>(1);
    WaitFlag<HardEvent::FIX_M>(0);
}

// ============================================================================
// 两层 K 循环：外层 kL1（GM→L1）+ 内层 kL0（L1→L0+Mmad）。
// 参考：cann-samples streamk/main.asc。
//   kL1 循环上 L1 双缓冲乒乓（l1BufId = kl1Idx & 1）：
//     MTE1_MTE2(0/1) —— GM→L1 加载同步；M_MTE1(0/1) —— L1→L0 拷贝同步。
//   kL0 循环上 L0 双缓冲乒乓（l0BufId = kl0Idx & 1）。
//   FIX_M(0) 是独立的跨 tile L0C 复用握手 flag，
//   不属于 L1/L0 双缓冲机制。
// 关键约束：SetFlag/WaitFlag 位置遵循 streamk —— 禁止重排或合并。
// ============================================================================

// GM → L1：将 A(curM, curKL1) + B(curKL1, curN) 的一个 kL1 块从 GM 加载到 L1。
// L1 张量由调用方（kL1 循环）创建并按引用传入，
// 使其在后续 kL0 循环切片时仍然存活。
// transB 完全在 GM 层处理：transB=1 时，GM B 声明为
// DNExt(k, n)（列主序）。CopyGM2L1 自动选择 DN2ZN，
// 在分形转换过程中完成转置。此处无需分支。
template <typename T, typename GmSliceA, typename GmSliceB, typename L1ATensor, typename L1BTensor>
__aicore__ inline void SpltGmToL1(
    const GmSliceA& gmBlockARow, const GmSliceB& gmBlockBCol,
    L1ATensor& tensorAL1, L1BTensor& tensorBL1,
    int32_t curM, int32_t curN, int32_t curKL1, int32_t kPosL1, int32_t l1BufId)
{
    WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);
    auto gmBlockA = gmBlockARow.Slice(Te::MakeCoord(0, kPosL1), Te::MakeShape(curM, curKL1));
    auto gmBlockB = gmBlockBCol.Slice(Te::MakeCoord(kPosL1, 0), Te::MakeShape(curKL1, curN));
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorAL1, gmBlockA);
    Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tensorBL1, gmBlockB);
    SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
    WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);
}

// 单步 L1→L0 + Mmad：从 L1 的 kOffL1 处切出 (curM, curKL0)，拷贝到 L0
// （经 l0BufId 偏移 = SPLT_HALF_L0_SIZE * l0BufId 双缓冲），再执行 Mmad。
// MTE1_M 同步确保 Mmad 前 L1→L0 拷贝已完成。M_MTE1 释放 L0。
template <typename T, typename L1ATensor, typename L1BTensor, typename L0CTensor>
__aicore__ inline void SpltL0AndMmadStep(
    const L1ATensor& tensorAL1, const L1BTensor& tensorBL1, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t curKL0, int32_t kOffL1,
    int32_t l0BufId, bool isFinalAcc, bool& cmatrixInitVal)
{
    WaitFlag<HardEvent::M_MTE1>(l0BufId);
    const int64_t l0Off = static_cast<int64_t>(SPLT_HALF_L0_SIZE) * l0BufId;
    auto tensorAL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, T>(l0Off),
                                Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(curM, curKL0));
    auto tensorBL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, T>(l0Off),
                                Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(curKL0, curN));
    auto tensorAL1Tile = tensorAL1.Slice(Te::MakeCoord(0, kOffL1), Te::MakeShape(curM, curKL0));
    auto tensorBL1Tile = tensorBL1.Slice(Te::MakeCoord(kOffL1, 0), Te::MakeShape(curKL0, curN));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}), tensorAL0, tensorAL1Tile);
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}), tensorBL0, tensorBL1Tile);
    SetFlag<HardEvent::MTE1_M>(l0BufId);
    WaitFlag<HardEvent::MTE1_M>(l0BufId);
    const uint8_t unitFlag = isFinalAcc ? static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION)
                                        : static_cast<uint8_t>(SPLT_NON_FINAL_ACCUMULATION);
    Te::MmadParams params{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                          static_cast<uint16_t>(curKL0), unitFlag, cmatrixInitVal};
    constexpr auto mmadAtom = Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{});
    Te::Mmad(mmadAtom.with(params), tensorL0C, tensorAL0, tensorBL0);
    SetFlag<HardEvent::M_MTE1>(l0BufId);
    cmatrixInitVal = false;
}

// 内层 kL0 循环：从 L1 缓冲按 baseK 大小的块迭代，每块执行 L1→L0 + Mmad。
// L0 双缓冲乒乓（l0BufId = kl0Idx & 1）。isFinalAcc 仅在最后一个 kL1
// 迭代的最后一个 kL0 迭代时为 true。
template <typename T, typename L1ATensor, typename L1BTensor, typename L0CTensor>
__aicore__ inline void SpltKL0Loop(
    const L1ATensor& tensorAL1, const L1BTensor& tensorBL1, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t curKL1, int32_t baseK,
    bool isLastKL1, bool& cmatrixInitVal)
{
    const int32_t kL0IterNum = (curKL1 + baseK - 1) / baseK;
    for (int32_t kl0Idx = 0; kl0Idx < kL0IterNum; ++kl0Idx) {
        const int32_t l0BufId = kl0Idx & 0x1;
        const int32_t curKL0 = (kl0Idx + 1 == kL0IterNum) ? (curKL1 - kl0Idx * baseK) : baseK;
        const int32_t kOffL1 = kl0Idx * baseK;
        const bool isFinalAcc = isLastKL1 && (kl0Idx + 1 == kL0IterNum);
        SpltL0AndMmadStep<T>(tensorAL1, tensorBL1, tensorL0C,
                             curM, curN, curKL0, kOffL1, l0BufId, isFinalAcc, cmatrixInitVal);
    }
}

// 外层 kL1 循环驱动：GM→L1（kL1 块）→ kL0 循环（baseK 分块）→ 释放 L1。
// L1 双缓冲乒乓（l1BufId = kl1Idx & 1）。kSegStart 对 K 位置施加偏移：
// splitK>1（非融合）时为 kSeg*kSegLen，splitK==1（融合）时为 0。
// 取代原先单层的 SpltKLoopDriver。
template <typename T, typename GmSliceA, typename GmSliceB, typename L0CTensor>
__aicore__ inline void SpltKL1Loop(
    const GmSliceA& gmBlockARow, const GmSliceB& gmBlockBCol, L0CTensor& tensorL0C,
    int32_t curM, int32_t curN, int32_t k, int32_t kSegLen, int32_t kSegStart,
    int32_t baseK, int32_t kL1Size, bool& cmatrixInitVal, const L1BufferConfig& l1cfg)
{
    const int32_t kL1TileNum = (kSegLen + kL1Size - 1) / kL1Size;
    for (int32_t kl1Idx = 0; kl1Idx < kL1TileNum; ++kl1Idx) {
        const int32_t l1BufId = kl1Idx & 0x1;
        const int32_t kPosL1 = kSegStart + kl1Idx * kL1Size;
        int32_t curKL1 = (kl1Idx + 1 == kL1TileNum) ? (kSegLen - kl1Idx * kL1Size) : kL1Size;
        if (kPosL1 + curKL1 > k) { curKL1 = k - kPosL1; }
        if (curKL1 <= 0) { break; }
        const bool isLastKL1 = (kl1Idx + 1 == kL1TileNum) || (kPosL1 + curKL1 >= k);
        auto tensorAL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, T>(l1cfg.aL1Off[l1BufId]),
                                    Te::MakeFrameLayout<Te::NZLayoutPtn, Te::LayoutTraitDefault<T>>(curM, curKL1));
        auto tensorBL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, T>(l1cfg.bL1Off[l1BufId]),
                                    Te::MakeFrameLayout<Te::ZNLayoutPtn, Te::LayoutTraitDefault<T>>(curKL1, curN));
        SpltGmToL1<T>(gmBlockARow, gmBlockBCol, tensorAL1, tensorBL1,
                      curM, curN, curKL1, kPosL1, l1BufId);
        SpltKL0Loop<T>(tensorAL1, tensorBL1, tensorL0C,
                       curM, curN, curKL1, baseK, isLastKL1, cmatrixInitVal);
        SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
    }
}

// ============================================================================
// 共享的 int32 域 splitK 累加，供 SpltEpilogueInt32AccPath（步骤 1-2）与
// SpltEpilogueChunkSplitKReduce 的 int32-temp 分支使用。
// 加载第一个 splitK 段到 accIntUB，再用向量 Add<int32_t> 累加其余段
// （原地操作：dst==src0；arch35/DAV_3510 支持）。
// 同步语义：
//   - 每次 DataCopyPad 加载后 MTE2_V 握手；
//   - 每次 Add 后 PipeBarrier<PIPE_V> + V_MTE2 握手 —— 在下一段
//     DataCopyPad 覆写 tempIntUB 前完成 WAR 防护。
// GmT 为 GM temp 元素类型（int32_t）；模板化使一个辅助函数同时服务
// GlobalTensor<int32_t> 与 GlobalTensor<TempType> 两类调用点。
// ============================================================================
template <typename GmT>
__aicore__ inline void SpltSplitKAccInt32(
    LocalTensor<int32_t>& accIntUB, LocalTensor<int32_t>& tempIntUB,
    GlobalTensor<GmT>& tempGM,
    int64_t base, int32_t count, int32_t splitK, int64_t totalElem,
    DataCopyExtParams& copyAcc, DataCopyPadExtParams<GmT>& padAcc)
{
    // 约束：两条调用路径（SpltEpilogueInt32AccPath 与 SpltEpilogueChunkSplitKReduce
    // 的 int32 分支）均以 ReinterpretCast<int32_t> 复用 float UB 后传入本函数，
    // 隐含 GmT == int32_t 的假设；SpltL0CTypeTrait 未来扩展新 temp 类型时，
    // 在此编译期拦截，防止静默出错。
    static_assert(std::is_same_v<GmT, int32_t>,
                  "SpltSplitKAccInt32 仅支持 int32_t 的 temp 元素类型");
    // 1. 加载第一个 splitK 段（int32 GM -> int32 UB）
    DataCopyPad(accIntUB, tempGM[base], copyAcc, padAcc);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // 2. 向量 int32 累加其余段
    for (int32_t s = 1; s < splitK; ++s) {
        DataCopyPad(tempIntUB, tempGM[static_cast<uint64_t>(s) * totalElem + base],
                    copyAcc, padAcc);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Add<int32_t>(accIntUB, accIntUB, tempIntUB, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

__aicore__ inline float SpltGetScalarFromUB(LocalTensor<float>& ubTensor, int32_t index)
{
    return ubTensor.GetValue(static_cast<uint32_t>(index));
}

// ============================================================================
// FastPath 判定共享谓词（B1）：fused RunTile 与 fused InitBuffers 共用；
// epilogue INT32 快速路径（SpltEpilogueTryInt32FastPath）同式。
// beta 向量缩放开启时 host 置 td.beta=0.0f 哨兵（betaVec 的 beta*C 由
// 向量分支承载），beta==0.0f 恒真——必须以 betaVectorScaling==0 显式排除
// (alphaVec=0, betaVec=1) 混合态，否则 FastPath 直接 Cast(acc)→D 会整体
// 丢失 beta*C。单一定义根除"RunTile 改判而 InitBuffers 未同步 → General
// Path Get 未分配 TBuf"的漂移风险。
// ============================================================================
__aicore__ inline bool SpltFastPathCond(float alpha, float beta,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    uint64_t biasDevPtr, int32_t activationType)
{
    return alphaVectorScaling == 0 && betaVectorScaling == 0
        && alpha == 1.0f && beta == 0.0f
        && biasDevPtr == 0 && activationType == 0;
}

// ============================================================================
// B5：alpha 的 int32 域整值判定。float→int32 收窄前显式域检查（C++ 语义
// 下超域 float→int32 转换为 UB）：[-2^31, 2^31) 内的 float 精确比较
// (alpha == (float)(int32_t)alpha) 判整值。超过 2^24 的整值 float 本身是
// 2 的幂间距（尾数 24bit），回转比较仍正确判定"可精确表示的整值"；不可精
// 确表示的非整值自动回落 FP32 域（调用方语义）。
// ============================================================================
__aicore__ inline bool SpltAlphaIsIntegralInt32(float alpha)
{
    return alpha >= -2147483648.0f && alpha < 2147483648.0f
        && alpha == static_cast<float>(static_cast<int32_t>(alpha));
}

// 将完整 bias 向量从 GM 加载到 FP32 UB（count 个元素）。
// BiasType=float（FP32/INT8）：直接 DataCopyPad 到 FP32 UB。
// BiasType=half（FP16）：DataCopyPad 原始数据到 biasRawBuf，再 Vector Cast 到 FP32 UB。
// BiasType=bfloat16_t（BF16）：DataCopyPad 原始数据到 biasRawBuf，再走 SpltCastBf16ToFp32Vec。
// BiasType != float 时 biasRawBuf 必须预先分配 count * sizeof(BiasType) 字节。
template <typename BiasType>
__aicore__ inline void SpltLoadBiasFullToFp32(
    LocalTensor<float>& biasFp32UB,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    GlobalTensor<BiasType>& biasGM,
    int32_t count)
{
    if constexpr (std::is_same_v<BiasType, float>) {
        // FP32/INT8 路径：GM 中 bias 为 FP32，直接加载
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padBias{false, 0, 0, 0.0f};
        DataCopyPad(biasFp32UB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        // FP16/BF16 路径：加载原始 bias，再 Vector Cast 到 FP32
        LocalTensor<BiasType> biasRawUB = biasRawBuf.Get<BiasType>();
        DataCopyExtParams copyBias{1, static_cast<uint32_t>(count * sizeof(BiasType)), 0, 0, 0};
        DataCopyPadExtParams<BiasType> padBias{false, 0, 0, BiasType(0)};
        DataCopyPad(biasRawUB, biasGM[0], copyBias, padBias);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
            // BF16：使用既有 Vector Cast 辅助函数（Cast<float, bfloat16_t, CAST_NONE>）
            SpltCastBf16ToFp32Vec(biasFp32UB, biasRawUB, count);
        } else {
            // FP16：直接 Vector Cast（Cast<float, half, CAST_NONE>）
            Cast(biasFp32UB, biasRawUB, AscendC::RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
    }
}

// 分配 bias UB 缓冲，并按需将完整 bias 向量从 GM 加载到 FP32 UB。
// 由 SpltEpilogueImpl（非融合）与 SpltFusedEpilogueImpl（融合）共享，
// 消除重复的 bias 初始化代码（#2）。两条路径构建相同的
// GlobalTensor + TBuf + LocalTensor + 条件调用 SpltLoadBiasFullToFp32。
// biasRawBuf 仅在 BiasType 非 float 时分配（FP16/BF16 的 Cast 中间缓冲）。
template <typename BiasType>
__aicore__ inline void SpltInitBiasBuffers(
    GlobalTensor<BiasType>& biasVecGM,
    TBuf<TPosition::VECCALC>& biasVecBuf,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB,
    TPipe& pipe,
    uint64_t biasDevPtr, int64_t biasStride, int32_t m)
{
    biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(biasDevPtr),
                               static_cast<uint64_t>(m));
    pipe.InitBuffer(biasVecBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)));
    biasVecUB = biasVecBuf.Get<float>();
    if constexpr (!std::is_same_v<BiasType, float>) {
        pipe.InitBuffer(biasRawBuf, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(BiasType)));
    }
    if (biasStride == 0) {
        // 所有 batch 共用同一 bias
        SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
    }
}

// C/D/bias 的逐 batch GM 基址设置（原非融合
// SpltEpilogueOffsetBatchGM 与融合 SpltFusedEpilogueOffsetBatchGM 的
// 重复段提取共享）：C/D 基址按 batchStrideC/D 偏移，长度按物理 span
// （ROW: (m-1)*ld+n / COL: (n-1)*ld+m，经 SpltCdSpanElems——ld>min / COL
// 时逻辑 m*n 小于实际物理跨度）；bias 基址按 biasStride 偏移。bias 的
// UB 重载条件（非融合恒全量 / 融合 chunkMode==0）由调用方决定，不在
// 本 helper。CType/OutType：C/D 实际 dtype；BiasType：bias dtype。
template <typename CType, typename OutType, typename BiasType>
__aicore__ inline void SpltOffsetBatchCdBiasGM(
    const AclsparseltTilingData& td,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<BiasType>& biasVecGM,
    GM_ADDR cGm, GM_ADDR dGm, int32_t b, int32_t m)
{
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(cGm)
                        + static_cast<int64_t>(b) * td.batchStrideC,
                        SpltCdSpanElems(m, td.n, td.ldC, td.cOrderIsCol != 0));
    dGM.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(dGm)
                        + static_cast<int64_t>(b) * td.batchStrideD,
                        SpltCdSpanElems(m, td.n, td.ldD, td.cOrderIsCol != 0));
    if (td.biasDevPtr != 0 && td.biasStride != 0) {
        biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(td.biasDevPtr)
                                  + static_cast<int64_t>(b) * td.biasStride,
                                  static_cast<uint64_t>(m));
    }
}

// 共享的缩放辅助函数：从 SpltEpilogueImpl 与 SpltFusedEpilogueImpl
// 提取，以消除重复代码（alphaVecGM/betaVecGM 构建与 UB 加载）。
// flag 操作完全保持不变。

// 向量缩放启用时，构建 alpha/beta 向量的 global buffer。
__aicore__ inline void SpltInitScalingGM(
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    uint64_t alphaDevPtr, uint64_t betaDevPtr, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling)
{
    if (alphaVectorScaling == 1) {
        alphaVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alphaDevPtr),
                                    static_cast<uint64_t>(m));
    }
    if (betaVectorScaling == 1) {
        betaVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(betaDevPtr),
                                   static_cast<uint64_t>(m));
    }
}

// 向量缩放启用时，分配 UB 并从 GM 加载 alpha/beta 向量。
// scalingChunkMode（= td.biasChunkMode）：1 时为 per-tile 惰性加载
// 模式——仅分配 chunkElems（= maxMAlign，单 tile 行数上界）的 FP32 槽位，
// 不在此处加载，由 SpltFusedEpilogueGeneralPath 逐 tile 加载
// alphaVecGM[mPos .. mPos+curM) 段（预算公式与降级动作对齐：向量族
// 全量加载超 UB 时三向量同走 chunk 策略，biasChunkMode 单标志控制）；
// 0 时维持全量加载（InitBuffer(m*4) + 立即 DataCopyPad，紧凑/小 m 域
// 零回归锚点路径）。two-stage（splitK>1）恒为 0（host 侧
// compute_bias_chunk_mode 仅在 splitK==1 置标志，超限域由
// validate_scaling_ub_budget 拒绝）。
__aicore__ inline void SpltLoadScalingUB(
    TBuf<TPosition::VECCALC>& alphaVecBuf, TBuf<TPosition::VECCALC>& betaVecBuf,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    TPipe& pipe, int32_t m,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    int32_t scalingChunkMode, uint32_t chunkElems)
{
    if (alphaVectorScaling == 1) {
        const uint32_t alphaElems = (scalingChunkMode == 1) ? chunkElems
            : static_cast<uint32_t>(static_cast<size_t>(m));
        pipe.InitBuffer(alphaVecBuf, alphaElems * sizeof(float));
        alphaVecUB = alphaVecBuf.Get<float>();
        if (scalingChunkMode == 0) {
            DataCopyExtParams copyAlpha{1, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padAlpha{false, 0, 0, 0.0f};
            DataCopyPad(alphaVecUB, alphaVecGM[0], copyAlpha, padAlpha);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }
    }
    if (betaVectorScaling == 1) {
        const uint32_t betaElems = (scalingChunkMode == 1) ? chunkElems
            : static_cast<uint32_t>(static_cast<size_t>(m));
        pipe.InitBuffer(betaVecBuf, betaElems * sizeof(float));
        betaVecUB = betaVecBuf.Get<float>();
        if (scalingChunkMode == 0) {
            DataCopyExtParams copyBeta{1, static_cast<uint32_t>(static_cast<size_t>(m) * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padBeta{false, 0, 0, 0.0f};
            DataCopyPad(betaVecUB, betaVecGM[0], copyBeta, padBeta);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }
    }
}

// ============================================================================
// bias + activation 的 epilogue 辅助函数。
// epilogue 链：SplitK reduce -> alpha -> beta*C -> bias -> activation -> Cast -> 写 D
// bias 和 activation 在 FP32 域（dFp32UB）上操作。
// ============================================================================

// 施加 ReLU 激活：D = min(upperBound, max(threshold, D))。
// 在 FP32 域（dFp32UB）操作。count 为元素数。
// 施加 ReLU 激活（归零语义，对齐 cuSPARSELt 官方 sample adv.cpp）：
//   D = (x <= threshold) ? 0 : min(x, upperBound)
// 旧实现 min(ub, max(thr, x)) 在 thr != 0 时把 x <= thr 域钳位到 thr，
// 与 sample 的归零语义不符；thr == 0（默认）时两语义全域逐位一致。
// 三步向量化（无标量逐元素循环）：
//   ① mask = (x > thr)   —— Compare 生成 b8 位掩码。Compare 的 count
//      形态在 3510 有 256B 对齐运行时断言（非对齐尾块不可用），故走
//      mask 参数形态 + Counter 模式（SetMaskCount 后 mask 参数即元素
//      总数，内部按 UpdateMask 精确处理任意长度的尾 repeat）。
//   ② y = min(x, ub)     —— Mins（保留旧实现的 ub 钳位）。
//   ③ out = mask ? y : 0 —— Select 模式 1（TENSOR_SCALAR，src1 为标量
//      0.0f，无需 zero 向量缓冲）；其 count 形态（Level 2）支持任意
//      元素数且无对齐断言。
// mask 对原始 x 比较而非对 min 后的 y 比较：thr > ub 的非常规组合下
// x > thr（此时必有 x > ub）应输出 min(x, ub) = ub 而非 0，与 sample
// 的逐语句顺序（先判 threshold 再 min ub）一致。
// thrUB 由调用方提供（>= count 个 float 的暂存，逐行/逐 piece 复用，
// 用于构造 threshold 常量向量——Compare 的 src1 为 tensor）；
// maskUB 为 b8 位掩码缓冲（>= (count+7)/8 字节，32B 对齐）。
__aicore__ inline void SpltApplyReLU(
    LocalTensor<float> dFp32UB, LocalTensor<float> thrUB,
    LocalTensor<uint8_t> maskUB,
    int32_t count, float reluThreshold, float reluUpperBound)
{
    Duplicate(thrUB, reluThreshold, count);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    BinaryRepeatParams reluBp;  // 默认构造：blkStride=1 / repStride=8（连续布局）
    Compare<float, uint8_t, false>(maskUB, dFp32UB, thrUB, CMPMODE::GT,
                                   static_cast<uint64_t>(count), 1, reluBp);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    Mins(dFp32UB, dFp32UB, reluUpperBound, count);
    PipeBarrier<PIPE_V>();
    Select(dFp32UB, maskUB, dFp32UB, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
}

// 施加 GeLU 激活（sigmoid 等价实现，11 步，含 Exp clamp）。
// 输入 x = dFp32UB，输出回写 dFp32UB。geluTemp 复用 tempBuf。
// 逐行执行（per-row, count=curN），geluTemp 需要一行大小 buffer。
// 步骤 6 clamp z<=20 防 Exp 溢出（不改变 golden 语义，sigmoid(20)≈1.0）。
// LocalTensor 按值传递（轻量级 handle，类似指针，避免临时对象无法绑定到非 const 引用）。
__aicore__ inline void SpltApplyGeLU(
    LocalTensor<float> dFp32UB, LocalTensor<float> geluTemp,
    int32_t count, float geluScaling)
{
    // 步骤 1：geluTemp = x²
    Mul(geluTemp, dFp32UB, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 2：geluTemp = 0.044715 * x²
    Muls(geluTemp, geluTemp, 0.044715f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 3：geluTemp = 0.044715 * x³
    Mul(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 4：geluTemp = x + 0.044715 * x³
    Add(geluTemp, geluTemp, dFp32UB, count);
    PipeBarrier<PIPE_V>();
    // 步骤 5：z = sqrt(8/π) * (x + 0.044715 * x³)
    Muls(geluTemp, geluTemp, 1.5957691f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 6：clamp z <= 20.0f 防 Exp 溢出
    Mins(geluTemp, geluTemp, 20.0f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 7：exp(z)
    Exp(geluTemp, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // 步骤 8：x = geluScaling * x
    Muls(dFp32UB, dFp32UB, geluScaling, count);
    PipeBarrier<PIPE_V>();
    // 步骤 9：x = geluScaling * x * exp(z)
    Mul(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
    // 步骤 10：geluTemp = 1 + exp(z)
    Adds(geluTemp, geluTemp, 1.0f, count);
    PipeBarrier<PIPE_V>();
    // 步骤 11：D = geluScaling * x * exp(z) / (1 + exp(z))
    Div(dFp32UB, dFp32UB, geluTemp, count);
    PipeBarrier<PIPE_V>();
}

#endif // SPLT_MATMUL_KERNEL_COMMON_H
