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
 * \file matmul_fused_kernel.cpp
 * \brief sparseLt 融合 matmul+epilogue kernel（__mix__(1,2)，splitK==1 路径）
 *        + extern "C" 启动器。
 * 超大源文件整改拆分单元（原 matmul_kernel.cpp，见
 * matmul_kernel.cpp 文件头）；共享基础设施位于 matmul_kernel_common.h。
 * kernel：
 *   - fused_matmul_kernel (__mix__(1,2)): splitK==1 时的 matmul+epilogue 融合
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
// FUSED MATMUL + EPILOGUE KERNEL（__mix__(1,2)）
//   splitK==1 路径：AIC 执行 matmul -> Fixpipe L0C->UB -> CrossCore 通知 AIV
//   AIV 执行 CrossCore 等待 -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
//   消除 GM temp 往返 + 1 次 kernel 启动。
//
// 架构：__mix__(1,2) = 每 block 1 个 AIC + 2 个 AIV。AIC 与 AIV 位于
// 不同物理核、各有独立 UB。CopyL0C2UB 依据 subBlockId 将完整 tile 数据
// 路由到目标 AIV 的 UB。tile 在 AIV0（subBlockId=false）与 AIV1
// （subBlockId=true）间交替以均衡负载。
// 简化起见不做乒乓（每 AIV 单 UB 槽位）；正确性优先。
// ============================================================================

// CrossCore 同步 flag 常量（块内 MODE 4）。
static constexpr uint8_t  SPLT_SYNC_MODE_4      = 4;    // 块内模式
static constexpr uint16_t SPLT_AIV0_SYNC_AIC    = 4;   // AIV0->AIC
static constexpr uint16_t SPLT_AIC_SYNC_AIV0    = 6;   // AIC->AIV0
static constexpr uint16_t SPLT_AIV1_SYNC_AIC    = 4 + 16;  // AIV1->AIC（FLAG_ID_MAX 偏移）
static constexpr uint16_t SPLT_AIC_SYNC_AIV1    = 6 + 16;  // AIC->AIV1

// ----------------------------------------------------------------------------
// AIC 侧：K 循环 + Fixpipe L0C->UB（subBlockId 交替）+ CrossCore
// ----------------------------------------------------------------------------
// SpltFusedMatmulCubeImpl 的 tile 循环驱动：m/n tile 遍历 + K 循环 +
// Fixpipe L0C->UB + CrossCore 同步。与 SpltMatmulCubeTileLoop 共享
// SpltKL1Loop。flag 操作保持不变。
template <typename T, typename GmTensorA, typename GmTensorB>
__aicore__ inline void SpltFusedMatmulCubeTileLoop(
    const AclsparseltTilingData& td,
    GmTensorA& gmA, GmTensorB& gmB,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalTiles, uint32_t& localTileIdx)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t baseK = td.baseK;
    const int32_t kSegLen = td.kSegLen;

    for (int32_t tileIdx = blockId; static_cast<int64_t>(tileIdx) < totalTiles; tileIdx += blockNum) {
        const int32_t mTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / nTiles);
        const int32_t nTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % nTiles);
        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);
        const uint64_t curMAlign = static_cast<uint64_t>((curM + 1) & ~1);
        // UB nAlign 必须用 L0C C0（=16），不能用 dtype C0（FP32 为 8）。
        // Fixpipe 将 L0C NZ 分形（C0=16）写入 UB NDExt。UB 中的行 stride
        // 为 CeilAlign(curN, 16) 个元素。若 curNAlign 使用 dtype C0（8），行 stride
        // 与 AIV 期望值（CeilAlign(curN, 16)）不匹配，导致非 16 对齐 curN 的
        // 数据错位（如 curN=36 -> AIC stride=40，AIV stride=48）。
        const uint64_t curNAlign = static_cast<uint64_t>((curN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;

        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B 切片对两条路径相同（DNExt GM 布局已处理 transB）。
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));
        // L0C 类型跟随 SpltL0CTypeTrait（FP32/FP16/BF16 为 float，INT8 为 int32_t）。
        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                     Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        WaitFlag<HardEvent::FIX_M>(0);
        bool cmatrixInitVal = true;
        // kSegStart=0：融合路径 splitK==1，无 K 段偏移。
        SpltKL1Loop<T>(gmBlockARow, gmBlockBCol, tensorL0C,
                       curM, curN, k, kSegLen, 0, baseK, td.kL1Size,
                       cmatrixInitVal, l1cfg);

        // ---- Fixpipe L0C -> UB（经 subBlockId 路由到目标 AIV）----
        // 交替规则：localTileIdx 偶数 -> AIV0（subBlockId=false），奇数 -> AIV1。
        const bool toAiv1 = (localTileIdx & 0x1) == 1;
        const uint16_t aivWaitFlag = toAiv1 ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
        const uint16_t aivNotifyFlag = toAiv1 ? SPLT_AIC_SYNC_AIV1 : SPLT_AIC_SYNC_AIV0;

        // 等待目标 AIV 释放 UB（反向依赖，首轮由 AIV 预置）。
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivWaitFlag);

        // UB 目的张量位于偏移 0（单槽位，无乒乓）。
        // Te::NDExtLayoutPtn 的 C0 必须与 L0C C0（DAV-3510 上 =16）一致，
        // 不能用 dtype C0（FP32 为 32/sizeof(float)=8）。不匹配会导致
        // Fixpipe 产生错误的分形拆分，输出错乱。
        // UB 张量类型跟随 L0C 类型（INT8 为 int32_t）。
        auto layoutUB = Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_L0C_C0>(curMAlign, curNAlign);
        auto ubTensor = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename SpltL0CTypeTrait<T>::type>(0), layoutUB);

        Te::FixpipeParams fixpParams;
        fixpParams.unitFlag = static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION);
        fixpParams.subBlockId = toAiv1;   // 路由到 AIV0 或 AIV1
        auto copyL0C2UB = Te::MakeCopy(Te::CopyL0C2UB{});
        Te::Copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);

        // 通知目标 AIV 数据已就绪。
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(aivNotifyFlag);

        SetFlag<HardEvent::FIX_M>(0);
        localTileIdx++;
    }
}

// SpltFusedMatmulCubeImpl 的 transB 分发：构造 gmB（依据 transB 选择
// DNExt 或 NDExt）并调用 SpltFusedMatmulCubeTileLoop。
// 消除重复的 transB 分支代码（4 个分支 -> 2 个 + 辅助函数）。
// B 的物理行距经 td.ldB 直通布局参数（dense B 描述符 ld 直读；B_pruned
// 恒紧凑 ldB==n；紧凑场景与旧逻辑维度参数逐位一致——零回归锚点），
// MTE2 搬运按行距跳过行尾 padding，非紧凑 dense B 无需预处理。
template <typename T, typename GmTensorA>
__aicore__ inline void SpltFusedMatmulCubeDispatchB(
    const AclsparseltTilingData& td, GmTensorA& gmA, GM_ADDR bGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalTiles, uint32_t& localTileIdx)
{
    const int32_t k = td.k;
    const int32_t n = td.n;
    if (td.transB != 0) {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(td.ldB, n));
        SpltFusedMatmulCubeTileLoop<T>(td, gmA, gmB, l1cfg, blockId, blockNum,
                                       nTiles, totalTiles, localTileIdx);
    } else {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, td.ldB));
        SpltFusedMatmulCubeTileLoop<T>(td, gmA, gmB, l1cfg, blockId, blockNum,
                                       nTiles, totalTiles, localTileIdx);
    }
}


template <typename T>
__aicore__ inline void SpltFusedMatmulCubeImpl(GM_ADDR aPrunedGm, GM_ADDR bGm,
                                               GM_ADDR dGm, GM_ADDR tilingGm)
{
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    // batch 参数
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;

    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (n + baseN - 1) / baseN;
    // 使用 int64_t 防止 mTiles*nTiles 接近
    // INT32_MAX（大 m、n）时溢出。与非融合路径一致。
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    if (totalTiles <= 0) { return; }

    // sparseTrans 处理：理由见 SpltMatmulCubeImpl。
    // A 的物理行距经 td.ldA 直通（A_pruned 恒紧凑 ldA==k 零回归锚点；
    // dense A 描述符 ld 直读，与 two-stage 路径一致）。
    // DNExt/NDExt 产生不同类型，因此 A 与 B 均需在调用点分支。
    const L1BufferConfig l1cfg = InitL1DoubleBuffer<T>(baseM, baseN, td.kL1Size);
    SetMMLayoutTransform(true);
    // batch 外循环（GM 指针按 batchStride 偏移）
    // localTileIdx 每 batch 重置为 0，与 AIV 侧 SpltFusedEpilogueTileLoop
    // 的 per-batch 重置对齐。此前 localTileIdx 在循环外声明并跨 batch 累加，
    // 导致 batch 边界处 AIC 奇偶翻转而 AIV 重置为 0，下一个 batch 首个 tile 的
    // AIC→AIV 路由 flag 与 AIV 等待 flag 错配 → CrossCore 死锁。
    for (int32_t b = 0; b < numBatches; ++b) {
        uint32_t localTileIdx = 0;  // 每 batch 重置，与 AIV 侧对齐
        // GM 指针按 batchStride 偏移（batchStride 以元素数为单位）
        __gm__ T* aBase = reinterpret_cast<__gm__ T*>(aPrunedGm)
                          + static_cast<int64_t>(b) * td.batchStrideA;
        GM_ADDR bBatchGm = bGm + static_cast<int64_t>(b) * td.batchStrideB * static_cast<int64_t>(sizeof(T));
        if (td.sparseTrans != 0) {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(td.ldA, k));
            SpltFusedMatmulCubeDispatchB<T>(td, gmA, bBatchGm, l1cfg, blockId, blockNum,
                                            nTiles, totalTiles, localTileIdx);
        } else {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, td.ldA));
            SpltFusedMatmulCubeDispatchB<T>(td, gmA, bBatchGm, l1cfg, blockId, blockNum,
                                            nTiles, totalTiles, localTileIdx);
        }
        // CrossCore drain 每 batch 执行：等待本 batch 各 AIV 释放 UB。
        // 若只 wait 最后一个 tile 的目标 AIV，另一路 AIV 的"最后释放" flag
        // 会残留置位；下一 batch 该 AIV 又在首个 tile 前预发一次释放
        // SetFlag（见 SpltFusedEpilogueTileLoop），两代 set 叠加使 AIC 在
        // 下一个 batch 的 tile WaitFlag 消费到旧 set —— AIC 在 AIV 尚未
        // 就绪时就 CopyL0C2UB 覆写 UB，破坏该 batch 首批 tile 的数据
        // （batch>1 且多 tile 时表现为部分 tile 首行数据陈旧）。按 blaze
        // ctor/dtor 预发排空模式，每 batch 结束把本 batch 实际参与的两路
        // AIV 的释放 flag 全部排空，保证 CrossCore set/wait 在 batch 边界
        // 严格配对。
        if (localTileIdx >= 2) {
            // 两路 AIV 都处理过 tile：排空 AIV0 与 AIV1 的最后释放。
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV0_SYNC_AIC);
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV1_SYNC_AIC);
        } else if (localTileIdx == 1) {
            // 仅 1 个 tile（localTileIdx=0 → 必为 AIV0）：只排空 AIV0。
            CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_FIX>(SPLT_AIV0_SYNC_AIC);
        }
    }

    // 排空 L1 双缓冲 flag。
    DrainCubeFlags();

    SetMMLayoutTransform(false);
}

// ----------------------------------------------------------------------------
// SpltFusedEpilogueImpl 的共享输出辅助函数：按 C/D 布局（ROW 批量/逐行
// DataCopyPad、COL 逐行 SpltCdColScatter）将 tile 结果写回 GM D。
// 模板化为 AccType（FP32/FP16/BF16 为 float，INT8 为 int32_t）与
// OutType（输出 dtype）。INT8 输出用 clamp+Cast；INT32 用 ROUND。
// 关键约束：SetFlag/WaitFlag 位置完全保持不变。
// ----------------------------------------------------------------------------

// FP32 输出：V_MTE3 同步 + 批量/逐行 DataCopyPad（OutType=float 时调用）。
// ROW 块起点经 SpltCdOffset（行步长 ldD）；批量分支的 GM 侧行间隔 =
// (ldD - curN)*elemSize（ld 接入），且 stride>0 时 blockCount 受 DAV-3510
// 2D 搬运 ~29-32 块隐式上限约束（api-datacopy.md），按 32 行分批。
// COL：批量分支不适用（物理连续段为逻辑列），走逐行 SpltCdColScatter。
__aicore__ inline void SpltFusedEpilogueOutputFp32(
    const LocalTensor<float>& srcUB, GlobalTensor<float>& dGM,
    LocalTensor<float>& wideCdUB, const SpltCdCtx& cd,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign)
{
    if (!cd.isCol && curN % 8 == 0) {
        const uint32_t srcStride = static_cast<uint32_t>(
            (curNAlign - curN) * sizeof(float) / 32);
        const int64_t dstStride = (static_cast<int64_t>(cd.ldD) - curN) * sizeof(float);
        // stride == 0（紧凑整列 tile）无隐式上限；stride > 0 按 32 行分批。
        const int32_t maxBatch = (dstStride > 0) ? SPLT_CD_COL_BLOCK : curM;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (int32_t r0 = 0; r0 < curM; r0 += maxBatch) {
            const int32_t rows = (curM - r0 > maxBatch) ? maxBatch : (curM - r0);
            DataCopyExtParams copyBatch{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(curN * sizeof(float)),
                srcStride, dstStride, 0};
            DataCopyPad(dGM[SpltCdOffset(mPos + r0, nPos, cd.ldD, false)], srcUB[r0 * curNAlign],
                        copyBatch);
        }
    } else if (!cd.isCol) {
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (int32_t r = 0; r < curM; ++r) {
            copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(float));
            DataCopyPad(dGM[SpltCdOffset(mPos + r, nPos, cd.ldD, false)],
                        srcUB[r * curNAlign], copyOut);
        }
    } else {
        // COL：逻辑行 r 的 curN 元素经元素块散开（块间 (ldD-1)*es）。
        // SpltCdColScatter 自带 V→标量→MTE3 的同步链。
        for (int32_t r = 0; r < curM; ++r) {
            SpltCdColScatter<float>(dGM, srcUB[r * curNAlign], wideCdUB,
                                    mPos + r, nPos, curN, cd.ldD);
        }
    }
}

// 非 FP32 输出：逐行 Cast + V_MTE3 + DataCopyPad + MTE3_V。
// INT8 输出 —— 先 Mins/Maxs clamp 再 Cast ROUND（CAST_SATURATE 不可用）。
// BF16 输出 —— Vector Cast<bfloat16_t, float, CAST_ROUND>。
// dFp32Buf 用作 INT8 clamp 的 float 中间缓冲（与输出 dTBuf 分开）。
// halfBuf 参数：float→int8_t 需要两步
// Cast（float→half→int8_t），因为 DAV-3510 不支持 float→int8 直转。
template <typename OutType>
__aicore__ inline void SpltFusedEpilogueOutputCast(
    const LocalTensor<float>& srcUB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<OutType>& dGM,
    LocalTensor<OutType>& wideCdUB, const SpltCdCtx& cd,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign)
{
    LocalTensor<OutType> dUB = dTBuf.Get<OutType>();
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    for (int32_t r = 0; r < curM; ++r) {
        if constexpr (std::is_same_v<OutType, int8_t>) {
            // INT8：float 先 clamp 到 [-128, 127]，再经 float→half→int8 Cast 为 int8。
            // DAV-3510 不支持 float→int8_t 直接
            // Cast；使用 SpltCastFp32ToInt8（两步 Vector Cast）。
            LocalTensor<float> clampUB = dFp32Buf.Get<float>();
            Mins(clampUB, srcUB[r * curNAlign], 127.0f, curN);
            Maxs(clampUB, clampUB, -128.0f, curN);
            SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        } else if constexpr (std::is_same_v<OutType, __bf16>) {
            // 向量化 FP32→BF16 Cast（取代标量循环）。
            // SpltCastFp32ToBf16Vec 直接使用 Vector Cast<bfloat16_t,float>；
            // 无需中间缓冲（dFp32Buf/halfBuf）。
            SpltCastFp32ToBf16Vec(dUB, srcUB[r * curNAlign], curN);
        } else {
            Cast(dUB, srcUB[r * curNAlign], AscendC::RoundMode::CAST_ROUND, curN);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(OutType));
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(mPos + r, nPos, cd.ldD, false)], dUB, copyOut);
        } else {
            SpltCdColScatter<OutType>(dGM, dUB, wideCdUB, mPos + r, nPos, curN, cd.ldD);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

// 融合 INT8→INT32 快速路径的 int32 UB→GM 直接输出。
// 无 Cast、无 float 中间量 —— 对大累加值保持精确的 int32 精度
// （float 在 [-2^24, 2^24] 之外损失精度）。
// 逻辑镜像 SpltFusedEpilogueOutputFp32（sizeof(int32_t)==sizeof(float)==4）。
__aicore__ inline void SpltFusedEpilogueOutputInt32Direct(
    const LocalTensor<int32_t>& srcUB, GlobalTensor<int32_t>& dGM,
    LocalTensor<int32_t>& wideCdUB, const SpltCdCtx& cd,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign)
{
    if (!cd.isCol && curN % 8 == 0) {
        const uint32_t srcStride = static_cast<uint32_t>(
            (curNAlign - curN) * sizeof(int32_t) / 32);
        const int64_t dstStride = (static_cast<int64_t>(cd.ldD) - curN) * sizeof(int32_t);
        const int32_t maxBatch = (dstStride > 0) ? SPLT_CD_COL_BLOCK : curM;
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (int32_t r0 = 0; r0 < curM; r0 += maxBatch) {
            const int32_t rows = (curM - r0 > maxBatch) ? maxBatch : (curM - r0);
            DataCopyExtParams copyBatch{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(curN * sizeof(int32_t)),
                srcStride, dstStride, 0};
            DataCopyPad(dGM[SpltCdOffset(mPos + r0, nPos, cd.ldD, false)], srcUB[r0 * curNAlign],
                        copyBatch);
        }
    } else if (!cd.isCol) {
        DataCopyExtParams copyOut{1, 0, 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (int32_t r = 0; r < curM; ++r) {
            copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(int32_t));
            DataCopyPad(dGM[SpltCdOffset(mPos + r, nPos, cd.ldD, false)],
                        srcUB[r * curNAlign], copyOut);
        }
    } else {
        for (int32_t r = 0; r < curM; ++r) {
            SpltCdColScatter<int32_t>(dGM, srcUB[r * curNAlign], wideCdUB,
                                      mPos + r, nPos, curN, cd.ldD);
        }
    }
}

// 融合 INT8→INT8 快速路径的向量化 int32→int8 clamp。
// 以 Vector 操作取代标量 GetValue/SetValue 循环（128×128 需 2.1ms）：
//   1. Cast int32→float（Vector，CAST_NONE —— [-2^24, 2^24] 内精确）
//   2. Mins/Maxs clamp 到 [-128, 127]（Vector）
//   3. SpltCastFp32ToInt8（float→half→int8，两步 Vector Cast）
// float 中间结果对 [-128,127] 内（clamp 后）的所有 int32 值精确，
// 且 Cast int32→float 在 [-2^24, 2^24] 内精确，覆盖所有实际
// 累加结果（alpha=1+beta=0 快速路径，无缩放）。
// flag 模式镜像 SpltFusedEpilogueOutputCast（逐行 V_MTE3 + MTE3_V）。
// dFp32Buf/halfBuf 已在 SpltFusedEpilogueImpl 中为 INT8 快速路径预分配。
__aicore__ inline void SpltFusedEpilogueOutputInt8VecClamp(
    const LocalTensor<int32_t>& srcUB, TBuf<TPosition::VECCALC>& dTBuf,
    TBuf<TPosition::VECCALC>& dFp32Buf, TBuf<TPosition::VECCALC>& halfBuf,
    GlobalTensor<int8_t>& dGM,
    LocalTensor<int8_t>& wideCdUB, const SpltCdCtx& cd,
    int32_t mPos, int32_t nPos, int32_t curM, int32_t curN,
    uint64_t curNAlign)
{
    LocalTensor<int8_t> dUB = dTBuf.Get<int8_t>();
    LocalTensor<float> clampUB = dFp32Buf.Get<float>();
    DataCopyExtParams copyOut{1, 0, 0, 0, 0};
    for (int32_t r = 0; r < curM; ++r) {
        // 1. 本行 Cast int32 → float
        Cast(clampUB, srcUB[r * curNAlign], AscendC::RoundMode::CAST_NONE, curN);
        PipeBarrier<PIPE_V>();
        // 2. clamp 到 [-128, 127]（先 Mins 后 Maxs，均为 Vector）
        Mins(clampUB, clampUB, 127.0f, curN);
        Maxs(clampUB, clampUB, -128.0f, curN);
        // 3. 经 half Cast float → int8（SpltCastFp32ToInt8 执行 float→half→int8）
        SpltCastFp32ToInt8(dUB, clampUB, halfBuf, curN);
        // 4. 本行输出到 GM（flag 模式与 SpltFusedEpilogueOutputCast 一致）
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        copyOut.blockLen = static_cast<uint32_t>(curN * sizeof(int8_t));
        if (!cd.isCol) {
            DataCopyPad(dGM[SpltCdOffset(mPos + r, nPos, cd.ldD, false)], dUB, copyOut);
        } else {
            SpltCdColScatter<int8_t>(dGM, dUB, wideCdUB, mPos + r, nPos, curN, cd.ldD);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

// 快速路径与 beta*C 段（SpltFusedEpilogueFastPath / SpltFusedEpilogueBetaC）
// 定义于 SpltFusedEpilogueTileCtx / SpltFusedEpilogueCfg 之后，直接以
// ctx/cfg 聚合参数接收句柄组与配置。

// ----------------------------------------------------------------------------
// 从 SpltFusedEpilogueGeneralPath 提取的子辅助函数。每个辅助函数保持精确的
// flag/pipe 操作。
// ----------------------------------------------------------------------------

// 对累加结果施加 alpha 缩放 -> dFp32UB。
// 同时处理标量 alpha 与逐行向量 alpha，覆盖 float 与
// int32 两种累加类型（INT8 需在 Muls 前 Cast int32->float）。
// scalingChunkMode==1（向量族 per-tile chunk 模式）时
// alphaVecUB 仅持有本 tile 的 [0, curM) 段（tile 段加载见
// SpltFusedEpilogueGeneralPath 开头），行索引取 r 而非 mPos + r；
// 0 时维持全量向量的 mPos + r 索引（零回归锚点路径）。
template <typename AccType>
__aicore__ inline void SpltFusedEpilogueApplyAlpha(
    LocalTensor<float>& dFp32UB, LocalTensor<AccType>& accUB,
    LocalTensor<float>& alphaVecUB,
    int32_t mPos, int32_t curM, int32_t curN,
    uint64_t curMAlign, uint64_t curNAlign,
    float alpha, int32_t alphaVectorScaling,
    int32_t scalingChunkMode)
{
    if (alphaVectorScaling == 1) {
        const int32_t baseIdx = (scalingChunkMode == 1) ? 0 : mPos;
        if constexpr (std::is_same_v<AccType, float>) {
            for (int32_t r = 0; r < curM; ++r) {
                float alpha_r = SpltGetScalarFromUB(alphaVecUB, baseIdx + r);
                Muls(dFp32UB[r * curNAlign], accUB[r * curNAlign], alpha_r, curN);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(dFp32UB, accUB, AscendC::RoundMode::CAST_NONE,
                 static_cast<int32_t>(curMAlign * curNAlign));
            PipeBarrier<PIPE_V>();
            for (int32_t r = 0; r < curM; ++r) {
                float alpha_r = SpltGetScalarFromUB(alphaVecUB, baseIdx + r);
                Muls(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], alpha_r, curN);
            }
            PipeBarrier<PIPE_V>();
        }
    } else {
        if constexpr (std::is_same_v<AccType, float>) {
            Muls(dFp32UB, accUB, alpha, static_cast<int32_t>(curMAlign * curNAlign));
        } else {
            // INT8：int32 acc 先 Cast -> float，再 Muls 乘 alpha。
            Cast(dFp32UB, accUB, AscendC::RoundMode::CAST_NONE,
                 static_cast<int32_t>(curMAlign * curNAlign));
            PipeBarrier<PIPE_V>();
            Muls(dFp32UB, dFp32UB, alpha, static_cast<int32_t>(curMAlign * curNAlign));
        }
    }
}

// 从 GM 加载非 FP32 的 bias chunk 并 Cast 到 FP32。
// 供 SpltFusedEpilogueApplyBias 用于 FP16/BF16 bias 类型。
template <typename BiasType>
__aicore__ inline void SpltCastBiasChunkToFp32(
    LocalTensor<BiasType>& biasRawChunkUB, LocalTensor<float>& biasChunkUB,
    GlobalTensor<BiasType>& biasVecGM, int32_t mPos, int32_t curM)
{
    DataCopyExtParams copyBiasChunk{1,
        static_cast<uint32_t>(curM * sizeof(BiasType)), 0, 0, 0};
    DataCopyPadExtParams<BiasType> padBiasChunk{false, 0, 0, BiasType(0)};
    DataCopyPad(biasRawChunkUB, biasVecGM[mPos], copyBiasChunk, padBiasChunk);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
        SpltCastBf16ToFp32Vec(biasChunkUB, biasRawChunkUB, curM);
    } else {
        Cast(biasChunkUB, biasRawChunkUB, AscendC::RoundMode::CAST_NONE, curM);
        PipeBarrier<PIPE_V>();
    }
}

// 逐行从 FP32 bias 向量取标量并 Adds 到 dFp32UB，完成后排空 V 队列。
// baseIdx：chunk 模式（向量仅持有本 tile 的 [0, curM) 段）取 0；
// 全量模式取 mPos（全量向量按全局行索引）。
__aicore__ inline void SpltAddBiasRows(LocalTensor<float>& dFp32UB,
    LocalTensor<float>& biasUB, int32_t curM, int32_t curN,
    uint64_t curNAlign, int32_t baseIdx)
{
    for (int32_t r = 0; r < curM; ++r) {
        float bias_r = SpltGetScalarFromUB(biasUB, baseIdx + r);
        Adds(dFp32UB[r * curNAlign], dFp32UB[r * curNAlign], bias_r, curN);
    }
    PipeBarrier<PIPE_V>();
}

// 融合通用路径中向 dFp32UB 施加 bias。
// 同时处理逐 chunk（biasChunkMode==1：从 GM 加载 curM 行）与
// 全量加载（biasChunkMode==0：biasVecUB 已加载）两种模式。
// bias 施加（bias dtype 恒键于 T=A/B dtype，与 C 的 CType 解耦）。
// FP16/BF16 bias 的原始数据在逐行 Adds 前先 Cast 到 FP32。
template <typename BiasType>
__aicore__ inline void SpltFusedEpilogueApplyBias(
    LocalTensor<float>& dFp32UB,
    TBuf<TPosition::VECCALC>& tempBuf, TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB, GlobalTensor<BiasType>& biasVecGM,
    int32_t mPos, int32_t curM, int32_t curN, uint64_t curNAlign,
    uint64_t biasDevPtr, int32_t biasChunkMode)
{
    if (biasDevPtr == 0) { return; }
    if (biasChunkMode == 1) {
        if constexpr (std::is_same_v<BiasType, float>) {
            // FP32/INT8：GM 中 bias 为 FP32，直接加载到 tempBuf
            LocalTensor<float> biasChunkUB = tempBuf.Get<float>();
            DataCopyExtParams copyBiasChunk{1,
                static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padBiasChunk{false, 0, 0, 0.0f};
            DataCopyPad(biasChunkUB, biasVecGM[mPos], copyBiasChunk, padBiasChunk);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            SpltAddBiasRows(dFp32UB, biasChunkUB, curM, curN, curNAlign, 0);
        } else {
            // FP16/BF16：加载原始 bias，Cast 到 FP32，逐行 Adds
            LocalTensor<BiasType> biasRawChunkUB = biasRawBuf.Get<BiasType>();
            LocalTensor<float> biasChunkUB = tempBuf.Get<float>();
            SpltCastBiasChunkToFp32<BiasType>(biasRawChunkUB, biasChunkUB, biasVecGM, mPos, curM);
            SpltAddBiasRows(dFp32UB, biasChunkUB, curM, curN, curNAlign, 0);
        }
    } else {
        // 全量加载：biasVecUB 已加载 (FP32)，按全局行索引逐行 Adds
        SpltAddBiasRows(dFp32UB, biasVecUB, curM, curN, curNAlign, mPos);
    }
}

// 融合通用路径中向 dFp32UB 施加激活（ReLU 或 GeLU）。
// ReLU / GeLU 均逐行执行，tempBuf 复用（一行大小 = maxNAlign×4）。
// ReLU 逐行化动机：归零语义的 Compare 需要 thr 常量向量（一行大小），
// 整 tile 一次调用需 tile 级 thr 缓冲（最大 ~136KB，UB 预算不可容纳）；
// 逐行复用与 GeLU 相同的一行缓冲，SpltApplyReLU 内部本就按向量宽度
// 分 repeat 处理，仅外层循环组织变化，指令数不变。
__aicore__ inline void SpltFusedEpilogueApplyActivation(
    LocalTensor<float>& dFp32UB, TBuf<TPosition::VECCALC>& tempBuf,
    LocalTensor<uint8_t>& maskUB,
    int32_t curM, int32_t curN, uint64_t curMAlign, uint64_t curNAlign,
    int32_t activationType,
    float reluThreshold, float reluUpperBound, float geluScaling)
{
    if (activationType == 1) {
        // ReLU: 逐行执行，thr 暂存复用 tempBuf（一行大小）
        LocalTensor<float> reluThrUB = tempBuf.Get<float>();
        for (int32_t r = 0; r < curM; ++r) {
            SpltApplyReLU(dFp32UB[r * curNAlign], reluThrUB, maskUB, curN,
                           reluThreshold, reluUpperBound);
        }
    } else if (activationType == 2) {
        // GeLU: 逐行执行，geluTemp 复用 tempBuf（一行大小 = maxNAlign×4）
        LocalTensor<float> geluTemp = tempBuf.Get<float>();
        for (int32_t r = 0; r < curM; ++r) {
            SpltApplyGeLU(dFp32UB[r * curNAlign], geluTemp, curN, geluScaling);
        }
    }
}

// ============================================================================
// 融合 epilogue 执行配置参数组。
// SpltFusedEpilogueGeneralPath 与 SpltFusedEpilogueTileLoop 此前各自携带
// 16 个相同的尾部参数（alpha/beta + 向量缩放 + bias + activation 配置
// 与句柄，两签名尾部逐行重复）——打包为本结构后两函数共享单一 cfg
// 参数。Local/GlobalTensor 按值成员（轻量 handle，仓内既有传递语义；
// 内部仅读写 tensor 指向的数据，不改写 handle 本身）；biasVecGM 类型
// 键于 BiasType，结构体模板化。由 splt_make_fused_epilogue_cfg 从 td
// 与已初始化的句柄组一次性构造。
// ============================================================================
template <typename BiasType>
struct SpltFusedEpilogueCfg {
    float alpha = 0.0f;
    float beta = 0.0f;
    int32_t alphaVectorScaling = 0;
    int32_t betaVectorScaling = 0;
    uint64_t biasDevPtr = 0;
    int32_t biasChunkMode = 0;
    int32_t activationType = 0;
    float reluThreshold = 0.0f;
    float reluUpperBound = 0.0f;
    float geluScaling = 1.0f;
    LocalTensor<float> alphaVecUB;
    LocalTensor<float> betaVecUB;
    GlobalTensor<float> alphaVecGM;
    GlobalTensor<float> betaVecGM;
    LocalTensor<float> biasVecUB;
    LocalTensor<uint8_t> maskUB;
    GlobalTensor<BiasType> biasVecGM;
};

// cfg 构造：标量配置取自 td，句柄组取自 Impl 已初始化的 UB/GM tensor。
template <typename BiasType>
__aicore__ inline SpltFusedEpilogueCfg<BiasType> splt_make_fused_epilogue_cfg(
    const AclsparseltTilingData& td,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    LocalTensor<float>& biasVecUB, LocalTensor<uint8_t>& maskUB,
    GlobalTensor<BiasType>& biasVecGM)
{
    SpltFusedEpilogueCfg<BiasType> cfg;
    cfg.alpha = td.alpha;
    cfg.beta = td.beta;
    cfg.alphaVectorScaling = td.alphaVectorScaling;
    cfg.betaVectorScaling = td.betaVectorScaling;
    cfg.biasDevPtr = td.biasDevPtr;
    cfg.biasChunkMode = td.biasChunkMode;
    cfg.activationType = td.activationType;
    cfg.reluThreshold = td.reluThreshold;
    cfg.reluUpperBound = td.reluUpperBound;
    cfg.geluScaling = td.geluScaling;
    cfg.alphaVecUB = alphaVecUB;
    cfg.betaVecUB = betaVecUB;
    cfg.alphaVecGM = alphaVecGM;
    cfg.betaVecGM = betaVecGM;
    cfg.biasVecUB = biasVecUB;
    cfg.maskUB = maskUB;
    cfg.biasVecGM = biasVecGM;
    return cfg;
}

// ============================================================================
// 融合 epilogue 逐 tile 参数组。
// 逐 tile 的 acc/C/D 句柄 + TBuf 引用组 + cd + tile 坐标打包为单一 ctx，
// GeneralPath / RunTile / FastPath / BetaC 共享。Local/GlobalTensor 按值
// 成员（轻量 handle，仓内既有传递语义；内部仅读写 tensor 指向的数据，
// 不改写 handle 本身）；TBuf 按引用绑定 TileLoop 归属的实体；cd 为
// 值拷贝（wideBuf 指针指向同一中转缓冲）；坐标为逐 tile 值。
// ============================================================================
template <typename AccType, typename OutType, typename CType = OutType>
struct SpltFusedEpilogueTileCtx {
    LocalTensor<AccType> accUB;
    GlobalTensor<CType> cGM;
    GlobalTensor<OutType> dGM;
    TBuf<TPosition::VECCALC>& dFp32Buf;
    TBuf<TPosition::VECCALC>& cTBuf;
    TBuf<TPosition::VECCALC>& cFp32Buf;
    TBuf<TPosition::VECCALC>& tempBuf;
    TBuf<TPosition::VECCALC>& dTBuf;
    TBuf<TPosition::VECCALC>& halfBuf;
    TBuf<TPosition::VECCALC>& biasRawBuf;
    SpltCdCtx cd;
    int32_t mPos;
    int32_t nPos;
    int32_t curM;
    int32_t curN;
    uint64_t curMAlign;
    uint64_t curNAlign;
};

// beta*C 累加循环（通用路径子步骤）：逐行加载 C（ROW DataCopyPad /
// COL SpltCdColGather），乘 beta（标量或逐行向量），累加到 dFp32UB。
// C dtype == OutType（validate_descriptors 强制 C 与 D 一致）。
// chunk 模式下 cfg.betaVecUB 仅持有本 tile 的 [0, curM) 段（tile 段加载
// 见 SpltFusedEpilogueGeneralPath 开头）。
template <typename AccType, typename OutType, typename CType, typename BiasType>
__aicore__ inline void SpltFusedEpilogueBetaC(
    SpltFusedEpilogueTileCtx<AccType, OutType, CType>& ctx,
    SpltFusedEpilogueCfg<BiasType>& cfg, LocalTensor<CType>& wideCcUB)
{
    LocalTensor<float> dFp32UB = ctx.dFp32Buf.template Get<float>();
    LocalTensor<CType> cUB = ctx.cTBuf.template Get<CType>();
    LocalTensor<float> tempUB = ctx.tempBuf.template Get<float>();
    DataCopyExtParams copyC{1, 0, 0, 0, 0};
    DataCopyPadExtParams<CType> padC{false, 0, 0, CType(0)};
    const int32_t betaBaseIdx = (cfg.biasChunkMode == 1) ? 0 : ctx.mPos;
    for (int32_t r = 0; r < ctx.curM; ++r) {
        copyC.blockLen = static_cast<uint32_t>(ctx.curN * sizeof(CType));
        if (!ctx.cd.isCol) {
            DataCopyPad(cUB,
                        ctx.cGM[SpltCdOffset(ctx.mPos + r, ctx.nPos, ctx.cd.ldC, false)],
                        copyC, padC);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        } else {
            SpltCdColGather<CType>(cUB, wideCcUB, ctx.cGM, ctx.mPos + r, ctx.nPos,
                                   ctx.curN, ctx.cd.ldC);
        }
        float beta_r = (cfg.betaVectorScaling == 1)
            ? SpltGetScalarFromUB(cfg.betaVecUB, betaBaseIdx + r) : cfg.beta;
        if constexpr (std::is_same_v<CType, float>) {
            Muls(tempUB, cUB, beta_r, ctx.curN);
        } else {
            LocalTensor<float> cFp32UB = ctx.cFp32Buf.template Get<float>();
            // BF16 使用向量化位操作 Cast（取代标量）。
            // tempBuf 复用为 Cast<uint32,uint16>+ShiftLeft 的 uint32 临时缓冲。
            // INT8 仍用标量 Cast：DAV-3510 Vector Cast
            // 不支持 int8_t→float（静默空操作，同 float→int8_t）。
            if constexpr (std::is_same_v<CType, __bf16>) {
                SpltCastBf16ToFp32Vec(cFp32UB, cUB, ctx.curN);
            } else if constexpr (std::is_same_v<CType, int8_t>) {
                SpltScalarCastLoop<float, CType>(cFp32UB, cUB, ctx.curN);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(cFp32UB, cUB, AscendC::RoundMode::CAST_NONE, ctx.curN);
            }
            Muls(tempUB, cFp32UB, beta_r, ctx.curN);
        }
        Add(dFp32UB[r * ctx.curNAlign], dFp32UB[r * ctx.curNAlign], tempUB, ctx.curN);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}

// 快速路径：alpha=1、beta=0、无 bias/activation -> acc 直接输出到 GM。
// FP32/FP16/BF16（AccType=float）：Cast（或直接）得到 OutType；
// INT8 累加（AccType=int32）采用 int32 直接处理（跳过 float Cast）：
//   - INT32 输出：int32 UB→GM 直接 DataCopyPad
//   - INT8 输出：向量化 Cast int32→float→clamp→int8
// INT8 的 Cast int32->float 必须覆盖整个 tile（curMAlign * curNAlign 个
// 元素），而不能只覆盖 curN（一行）——用 curN 会使 accFp32 第 1 行及
// 以后未初始化。halfBuf 用于 float→half→int8 两步 Cast。
template <typename AccType, typename OutType, typename CType = OutType>
__aicore__ inline void SpltFusedEpilogueFastPath(
    SpltFusedEpilogueTileCtx<AccType, OutType, CType>& ctx)
{
    LocalTensor<OutType> wideCdUB = (ctx.cd.wideBuf != nullptr)
        ? ctx.cd.wideBuf->template Get<OutType>() : LocalTensor<OutType>{};
    if constexpr (std::is_same_v<AccType, float>) {
        // FP32/FP16/BF16：acc 为 float，Cast（或直接）得到 OutType。
        if constexpr (std::is_same_v<OutType, float>) {
            SpltFusedEpilogueOutputFp32(ctx.accUB, ctx.dGM, wideCdUB, ctx.cd,
                                        ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                        ctx.curNAlign);
        } else {
            SpltFusedEpilogueOutputCast<OutType>(ctx.accUB, ctx.dTBuf, ctx.dFp32Buf,
                                                 ctx.halfBuf, ctx.dGM,
                                                 wideCdUB, ctx.cd,
                                                 ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                                 ctx.curNAlign);
        }
    } else {
        // INT8 快速路径：int32 直接处理，跳过 float Cast。
        if constexpr (std::is_same_v<OutType, int32_t>) {
            // INT32 输出：int32 UB→GM 直接写（无 Cast、无 float 中间量）
            SpltFusedEpilogueOutputInt32Direct(ctx.accUB, ctx.dGM, wideCdUB, ctx.cd,
                                               ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                               ctx.curNAlign);
        } else {
            // INT8 输出：向量化 int32→float→clamp→int8（取代标量循环）
            SpltFusedEpilogueOutputInt8VecClamp(ctx.accUB, ctx.dTBuf, ctx.dFp32Buf,
                                                ctx.halfBuf, ctx.dGM,
                                                wideCdUB, ctx.cd,
                                                ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                                ctx.curNAlign);
        }
    }
}

// alpha/beta 缩放向量的逐 tile chunk 惰性加载。
// 向量族 chunk 模式（biasChunkMode==1）：缩放向量逐 tile 惰性加载
// 本 tile 行段 [mPos, mPos+curM)（与 bias 的 per-chunk 加载同范式）。
// GM 源偏移 mPos*4 为 baseM*4 的整数倍（baseM 各档 >= 32），基址 32B
// 对齐（host 侧 validate_vector_scaling_pointers 校验）下段源恒 32B
// 对齐；目的 vecUB 为 maxMAlign*4 的 chunk 槽位（SpltLoadScalingUB
// chunk 分支分配），curM <= baseM <= maxMAlign 不越界。alpha/beta 两个
// 调用点经参数复用同一实现（字段对称，非重复块）。
__aicore__ inline void SpltFusedEpilogueLoadScalingVecChunk(
    LocalTensor<float>& vecUB, GlobalTensor<float>& vecGM,
    int32_t mPos, int32_t curM)
{
    DataCopyExtParams copyVecChunk{1,
        static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padVecChunk{false, 0, 0, 0.0f};
    DataCopyPad(vecUB, vecGM[mPos], copyVecChunk, padVecChunk);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}

// 通用路径：D = alpha*acc + beta*C + bias + activation -> 输出。
// INT8 路径（AccType=int32_t）需在 Muls 前 Cast int32 -> float。
// halfBuf 用于 float→half→int8 两步 Cast。
// bias + activation 插入在 beta*C 之后、Cast+output 之前。
// 配置经 SpltFusedEpilogueCfg、句柄与 tile 坐标经
// SpltFusedEpilogueTileCtx 打包传入。
template <typename AccType, typename OutType, typename CType = OutType, typename BiasType = float>
__aicore__ inline void SpltFusedEpilogueGeneralPath(
    SpltFusedEpilogueTileCtx<AccType, OutType, CType>& ctx,
    SpltFusedEpilogueCfg<BiasType>& cfg)
{
    LocalTensor<float> dFp32UB = ctx.dFp32Buf.template Get<float>();
    LocalTensor<OutType> wideCdUB = (ctx.cd.wideBuf != nullptr)
        ? ctx.cd.wideBuf->template Get<OutType>() : LocalTensor<OutType>{};
    LocalTensor<CType> wideCcUB = (ctx.cd.wideBuf != nullptr)
        ? ctx.cd.wideBuf->template Get<CType>() : LocalTensor<CType>{};
    // 向量族 chunk 模式：缩放向量逐 tile 惰性加载（对齐/越界论证
    // 见 SpltFusedEpilogueLoadScalingVecChunk）。
    if (cfg.biasChunkMode == 1 && cfg.alphaVectorScaling == 1) {
        SpltFusedEpilogueLoadScalingVecChunk(cfg.alphaVecUB, cfg.alphaVecGM, ctx.mPos, ctx.curM);
    }
    if (cfg.biasChunkMode == 1 && cfg.betaVectorScaling == 1) {
        SpltFusedEpilogueLoadScalingVecChunk(cfg.betaVecUB, cfg.betaVecGM, ctx.mPos, ctx.curM);
    }
    SpltFusedEpilogueApplyAlpha<AccType>(dFp32UB, ctx.accUB, cfg.alphaVecUB,
        ctx.mPos, ctx.curM, ctx.curN, ctx.curMAlign, ctx.curNAlign, cfg.alpha,
        cfg.alphaVectorScaling, cfg.biasChunkMode);
    if (cfg.betaVectorScaling == 1 || cfg.beta != 0.0f) {
        SpltFusedEpilogueBetaC<AccType, OutType, CType, BiasType>(ctx, cfg, wideCcUB);
    }
    // ③ bias → 累加到 dFp32UB
    SpltFusedEpilogueApplyBias<BiasType>(dFp32UB, ctx.tempBuf, ctx.biasRawBuf,
                                         cfg.biasVecUB, cfg.biasVecGM,
        ctx.mPos, ctx.curM, ctx.curN, ctx.curNAlign, cfg.biasDevPtr, cfg.biasChunkMode);
    // ④ activation
    SpltFusedEpilogueApplyActivation(dFp32UB, ctx.tempBuf, cfg.maskUB, ctx.curM, ctx.curN,
                                     ctx.curMAlign,
                                     ctx.curNAlign, cfg.activationType, cfg.reluThreshold,
                                     cfg.reluUpperBound, cfg.geluScaling);
    if constexpr (std::is_same_v<OutType, float>) {
        SpltFusedEpilogueOutputFp32(dFp32UB, ctx.dGM, wideCdUB, ctx.cd,
                                    ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                    ctx.curNAlign);
    } else {
        SpltFusedEpilogueOutputCast<OutType>(dFp32UB, ctx.dTBuf, ctx.dFp32Buf, ctx.halfBuf,
                                            ctx.dGM,
                                            wideCdUB, ctx.cd,
                                            ctx.mPos, ctx.nPos, ctx.curM, ctx.curN,
                                            ctx.curNAlign);
    }
}

// ----------------------------------------------------------------------------
// AIV 侧：CrossCore 等待 -> Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM
// 每个 AIV 处理分派给自己的 tile（与另一 AIV 交替）。
// ----------------------------------------------------------------------------

// 单 tile 的 FastPath/GeneralPath 分派。
//   fastPathEligible && fastPathCond -> FastPath；其余 -> GeneralPath。
// FastPath 条件：有 bias 或 activation 时走 GeneralPath（方式 A）。
// INT8 累加（AccType=int32）的快速路径仅支持 INT8/INT32 输出；
// INT8 A/B × C/D=FP16/BF16（超集组合）恒走 GeneralPath（int32→float
// Cast 后由既有 OutputCast 链写出）。编译期守卫保证 FastPath 模板
// 不会以 (int32 acc, half/bf16 out) 组合实例化（其内部 INT8 输出
// 分支无该组合的实例化路径）。
// 句柄与 tile 坐标经 SpltFusedEpilogueTileCtx 打包，
// 与 SpltFusedEpilogueGeneralPath 共享同一 ctx 形参。
template <typename AccType, typename OutType, typename CType, typename BiasType>
__aicore__ inline void SpltFusedEpilogueRunTile(
    SpltFusedEpilogueTileCtx<AccType, OutType, CType>& ctx,
    SpltFusedEpilogueCfg<BiasType>& cfg)
{
    constexpr bool fastPathEligible = std::is_same_v<AccType, float>
        || std::is_same_v<OutType, int8_t> || std::is_same_v<OutType, int32_t>;
    const bool fastPathCond = (cfg.alphaVectorScaling == 0 && cfg.alpha == 1.0f
                               && cfg.beta == 0.0f
                               && cfg.biasDevPtr == 0 && cfg.activationType == 0);
    bool ranFastPath = false;
    if constexpr (fastPathEligible) {
        if (fastPathCond) {
            // 快速路径：alpha=1、beta=0、无 bias、无 activation -> Cast acc -> OutType 并写 GM。
            SpltFusedEpilogueFastPath(ctx);
            ranFastPath = true;
        }
    }
    if (!ranFastPath) {
        // 通用路径：D = alpha*acc + beta*C + bias + activation。
        SpltFusedEpilogueGeneralPath<AccType, OutType, CType, BiasType>(ctx, cfg);
    }
}

// SpltFusedEpilogueImpl 的 tile 循环驱动：逐 tile 的
// CrossCore 等待 -> FastPath/GeneralPath -> CrossCore 通知遍历。
// flag 操作完全保持不变。TBufs/TPipe 归调用方（Impl）所有并按引用传入；
// 其生命周期覆盖整个循环。
// OutType 为输出 dtype（非 INT8 时等于 T；INT8 时为 int8_t/int32_t）。
// BiasType：bias dtype（恒 = SpltBiasDType<T>，由顶层 Impl 推导传入——
// bias 与 C 的 dtype 解耦：C 键于 CType、bias 键于 T）。
// 配置经 SpltFusedEpilogueCfg 打包；
// FastPath/GeneralPath 分派段见 SpltFusedEpilogueRunTile。
template <typename AccType, typename OutType, typename CType = OutType, typename BiasType = float>
__aicore__ inline void SpltFusedEpilogueTileLoop(
    const AclsparseltTilingData& td,
    const SpltCdCtx& cd,
    int32_t rawBlockId, int32_t blockNum, uint32_t subBlockIdx,
    uint16_t aivWaitFlag, uint16_t aivNotifyFlag,
    int32_t nTiles, int64_t totalTiles,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf, TBuf<TPosition::VECCALC>& biasRawBuf,
    SpltFusedEpilogueCfg<BiasType>& cfg)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;

    uint32_t localTileIdx = 0;
    bool ubReleased = false;  // 逐 batch 的首次释放标记
    for (int32_t tileIdx = rawBlockId; static_cast<int64_t>(tileIdx) < totalTiles; tileIdx += blockNum) {
        const bool myTile = (localTileIdx & 0x1) == subBlockIdx;

        const int32_t mTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / nTiles);
        const int32_t nTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % nTiles);
        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);
        const uint64_t curMAlign = static_cast<uint64_t>((curM + 1) & ~1);
        const uint64_t curNAlign = static_cast<uint64_t>((curN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;

        // 仅对本 AIV 分派到的 tile 参与同步。
        // AIC 只通知并等待目标 AIV。
        if (!myTile) {
            localTileIdx++;
            continue;
        }

        // 在本 AIV 于该 batch 的首个 tile 前先释放一次。仅当本 AIV
        // 确实拥有 tile 时才释放，使 CrossCore set/wait 对在 batch 边界
        // 严格配对（某 batch 内 tile 数为零的 AIV 不会发出多余的释放，
        // 导致 AIC 下一 batch 的 wait 错序消费）。
        if (!ubReleased) {
            CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
            ubReleased = true;
        }

        // 1. 等待 AIC Fixpipe 完成（数据已就绪于 UB）。
        CrossCoreWaitFlag<SPLT_SYNC_MODE_4, PIPE_V>(aivWaitFlag);

        // 2. 处理整个 tile（本 AIV 处理全部 curM 行）。
        LocalTensor<AccType> accUB = accBuf.Get<AccType>();

        // 逐 tile 参数组一次构造（句柄引用 + 坐标值）。
        // FastPath/GeneralPath 分派。
        SpltFusedEpilogueTileCtx<AccType, OutType, CType> tileCtx{
            accUB, cGM, dGM, dFp32Buf, cTBuf, cFp32Buf, tempBuf, dTBuf, halfBuf,
            biasRawBuf, cd, mPos, nPos, curM, curN, curMAlign, curNAlign};
        SpltFusedEpilogueRunTile<AccType, OutType, CType, BiasType>(tileCtx, cfg);

        // 3. 通知 AIC：本 AIV 已用完该 UB 槽位。
        CrossCoreSetFlag<SPLT_SYNC_MODE_4, PIPE_MTE3>(aivNotifyFlag);
        localTileIdx++;
    }
}

// SpltFusedEpilogueImpl 的 TBuf 初始化（按 tile 分配）。
// CType：C 矩阵实际 dtype——cTBuf 按 max(sizeof(T), sizeof(CType)) 分配
// （合法域内恒有 sizeof(CType) >= sizeof(T)，取 max 为防御性写法），
// cFp32Buf 条件扩展为 CType != float（C=INT32/INT8/FP16/BF16 均需
// Cast 到 FP32 的中间缓冲）。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltFusedEpilogueInitBuffers(
    TPipe& pipe,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf,
    TBuf<TPosition::VECCALC>& wideCdBuf, bool cOrderIsCol,
    uint64_t maxMAlign, uint64_t maxNAlign,
    float alpha, float beta,
    int32_t alphaVectorScaling, int32_t betaVectorScaling,
    uint64_t biasDevPtr, int32_t biasChunkMode, int32_t activationType)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    // FastPath 条件——有 bias 或 activation 时走 GeneralPath
    const bool isFastPath = (alphaVectorScaling == 0 && alpha == 1.0f && beta == 0.0f
                             && biasDevPtr == 0 && activationType == 0);
    pipe.InitBuffer(accBuf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(AccType)));
    if (!isFastPath) {
        pipe.InitBuffer(dFp32Buf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(float)));
    }
    if (isFastPath && !std::is_same_v<AccType, float>) {
        pipe.InitBuffer(dFp32Buf, static_cast<uint32_t>(maxMAlign * maxNAlign * sizeof(float)));
    }
    // 移除 BF16 快速路径的 dFp32Buf 分配：SpltCastFp32ToBf16Vec
    // （BF16 时在 SpltFusedEpilogueOutputCast 中调用）不使用 dFp32Buf。
    if constexpr (!std::is_same_v<OutType, float>) {
        pipe.InitBuffer(dTBuf, static_cast<uint32_t>(maxNAlign * sizeof(OutType)));
    }
    if constexpr (std::is_same_v<OutType, int8_t>) {
        pipe.InitBuffer(halfBuf, static_cast<uint32_t>(maxNAlign * sizeof(half)));
    }
    // X1：COL 中转缓冲（每元素一个 32B 槽位，容量 = tile 内单行最大段长）。
    if (cOrderIsCol) {
        pipe.InitBuffer(wideCdBuf, static_cast<uint32_t>(maxNAlign)
                                        * static_cast<uint32_t>(SPLT_CD_COL_BLOCK_BYTES));
    }
    // tempBuf/cTBuf/cFp32Buf 分配条件扩展
    // tempBuf: beta*C 中间量 | biasChunkUB（biasChunkMode==1，dtype=BiasType）|
    //          geluTemp（activationType==2）| ReLU thr 常量向量（activationType==1，
    //          逐行复用一行大小缓冲）
    // cTBuf: C 加载（beta!=0），按 CType 元素大小
    // cFp32Buf: C Cast 中间缓冲（beta!=0，CType!=float）
    const bool needTempBuf = (betaVectorScaling == 1 || beta != 0.0f)
        || (biasDevPtr != 0 && biasChunkMode == 1)
        || (activationType == 1)
        || (activationType == 2);
    const bool needCTBuf = (betaVectorScaling == 1 || beta != 0.0f);
    if (needCTBuf) {
        const uint32_t cElemSize = (sizeof(T) > sizeof(CType)) ? sizeof(T) : sizeof(CType);
        pipe.InitBuffer(cTBuf, static_cast<uint32_t>(maxNAlign * cElemSize));
    }
    if (needTempBuf) {
        pipe.InitBuffer(tempBuf, static_cast<uint32_t>(maxNAlign * sizeof(float)));
    }
    if ((betaVectorScaling == 1 || beta != 0.0f) && !std::is_same_v<CType, float>) {
        pipe.InitBuffer(cFp32Buf, static_cast<uint32_t>(maxNAlign * sizeof(float)));
    }
}

// [附注1] 原 SpltFusedEpilogueInitGM（以逻辑 m*n 作为 cGM/dGM 的
// SetGlobalBuffer 长度）为无调用死代码——fused 路径的 cGM/dGM 实际在
// SpltFusedEpilogueOffsetBatchGM 的 batch 循环（b=0 起）初始化，已删除；
// 长度口径统一为物理 span（见下方 SpltCdSpanElems 接入）。

// 为 SpltFusedEpilogueBatchLoop 设置逐 batch 的 GM 指针。
// 将 c/d/bias GM 按 batch stride 偏移；biasChunkMode==0 时重载完整 bias。
// CType：C 矩阵实际 dtype（batchStrideC 以 CType 元素为单位推进——
// kernel 既有类型化指针偏移语义自动正确）。bias 恒键于 T（INT8 A/B 恒
// FP32 bias，NVIDIA 规则）。
// C/D/bias 的 batch 基址段经 SpltOffsetBatchCdBiasGM
// 共享（非融合侧 SpltEpilogueOffsetBatchGM 同款）。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltFusedEpilogueOffsetBatchGM(
    const AclsparseltTilingData& td,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    GM_ADDR cGm, GM_ADDR dGm, int32_t b, int32_t m)
{
    using BiasType = SpltBiasDType<T>;
    // [附注1] C/D 长度按物理 span（SpltCdSpanElems）——ld>min / COL 时逻辑
    // m*n 小于实际物理跨度；C/D 同 order（描述符层一致性）。
    SpltOffsetBatchCdBiasGM<CType, OutType, BiasType>(td, cGM, dGM, biasVecGM,
                                                      cGm, dGm, b, m);
    // 融合路径 chunk 模式（biasChunkMode==1）bias 驻留 GM，仅全量模式重载。
    if (td.biasDevPtr != 0 && td.biasStride != 0 && td.biasChunkMode == 0) {
        SpltLoadBiasFullToFp32<BiasType>(biasVecUB, biasRawBuf, biasVecGM, m);
    }
}

// SpltFusedEpilogueImpl 的 batch 循环：遍历 batch，设置逐 batch GM
// 指针，按需重载 bias，并执行 tile 处理循环。
// flag 操作与 CrossCore 同步完全保持不变。
// alpha/beta/缩放/bias/activation 的配置与句柄组由 SpltFusedEpilogueCfg
// 一次性构造传递。
template <typename T, typename CType, typename OutType>
__aicore__ inline void SpltFusedEpilogueBatchLoop(
    const AclsparseltTilingData& td,
    const SpltCdCtx& cd,
    GlobalTensor<CType>& cGM, GlobalTensor<OutType>& dGM,
    GlobalTensor<SpltBiasDType<T>>& biasVecGM,
    LocalTensor<float>& biasVecUB, TBuf<TPosition::VECCALC>& biasRawBuf,
    TBuf<TPosition::VECCALC>& accBuf, TBuf<TPosition::VECCALC>& dFp32Buf,
    TBuf<TPosition::VECCALC>& dTBuf, TBuf<TPosition::VECCALC>& cTBuf,
    TBuf<TPosition::VECCALC>& cFp32Buf, TBuf<TPosition::VECCALC>& tempBuf,
    TBuf<TPosition::VECCALC>& halfBuf,
    LocalTensor<float>& alphaVecUB, LocalTensor<float>& betaVecUB,
    GlobalTensor<float>& alphaVecGM, GlobalTensor<float>& betaVecGM,
    LocalTensor<uint8_t>& maskUB,
    GM_ADDR cGm, GM_ADDR dGm)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    const int32_t m = td.m;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;
    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const uint32_t taskRation = GetTaskRation();
    const int32_t rawBlockId = blockId / static_cast<int32_t>(taskRation);
    const uint32_t subBlockIdx = GetSubBlockIdx();
    const uint16_t aivWaitFlag = (subBlockIdx == 1) ? SPLT_AIC_SYNC_AIV1 : SPLT_AIC_SYNC_AIV0;
    const uint16_t aivNotifyFlag = (subBlockIdx == 1) ? SPLT_AIV1_SYNC_AIC : SPLT_AIV0_SYNC_AIC;
    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (td.n + baseN - 1) / baseN;
    const int64_t totalTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    // cfg 按 Modifiable 引用传递：下游 ApplyAlpha/BetaC/ApplyBias/
    // ApplyActivation 以非 const LocalTensor/GlobalTensor 引用接收句柄。
    SpltFusedEpilogueCfg<SpltBiasDType<T>> cfg =
        splt_make_fused_epilogue_cfg<SpltBiasDType<T>>(
            td, alphaVecUB, betaVecUB, alphaVecGM, betaVecGM, biasVecUB, maskUB, biasVecGM);

    for (int32_t b = 0; b < numBatches; ++b) {
        SpltFusedEpilogueOffsetBatchGM<T, CType, OutType>(td, cGM, dGM, biasVecGM,
                                                   biasVecUB, biasRawBuf,
                                                   cGm, dGm, b, m);
        // 无 tile 的 block 跳过 tile 循环（也不参与任何 CrossCore 交互）。
        // 预释放 SetFlag 由 SpltFusedEpilogueTileLoop 在首个 myTile 前按需
        // 预发，见该函数内注释。
        if (static_cast<int64_t>(rawBlockId) >= totalTiles) { continue; }
        SpltFusedEpilogueTileLoop<AccType, OutType, CType, SpltBiasDType<T>>(td, cd,
                                     rawBlockId, blockNum, subBlockIdx,
                                     aivWaitFlag, aivNotifyFlag,
                                     nTiles, totalTiles, cGM, dGM,
                                     accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf, biasRawBuf,
                                     cfg);
    }
}

// SpltFusedEpilogueImpl 的 bias 缓冲初始化，按 biasChunkMode 分发。
//   biasChunkMode == 0：经 SpltInitBiasBuffers 全量加载 bias（驻留 UB）。
//   biasChunkMode == 1：chunk 模式 bias 驻留 GM（逐 tile 惰性加载），
//                       仅分配 Cast 中间缓冲（biasRawBuf）。
template <typename BiasType>
__aicore__ inline void SpltFusedEpilogueInitBias(
    GlobalTensor<BiasType>& biasVecGM,
    TBuf<TPosition::VECCALC>& biasVecBuf,
    TBuf<TPosition::VECCALC>& biasRawBuf,
    LocalTensor<float>& biasVecUB,
    TPipe& pipe,
    uint64_t biasDevPtr, int64_t biasStride, int32_t biasChunkMode,
    int32_t m, uint64_t maxMAlign)
{
    if (biasDevPtr == 0) { return; }
    if (biasChunkMode == 0) {
        SpltInitBiasBuffers<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                       pipe, biasDevPtr, biasStride, m);
    } else if (biasChunkMode == 1) {
        biasVecGM.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(biasDevPtr),
                                   static_cast<uint64_t>(m));
        if constexpr (!std::is_same_v<BiasType, float>) {
            pipe.InitBuffer(biasRawBuf, static_cast<uint32_t>(maxMAlign * sizeof(BiasType)));
        }
    }
}

// Impl 早退守卫（taskRation / totalTiles）：taskRation==0（无 Vector
// 子核分派，防零除）或 totalTiles<=0（空 tile 域）时本核直接返回。
// baseM/baseN 参与除法前显式非零守卫：二者来自 host
// resolve_cube_tiling（各档 >= 32，正常恒 > 0）；异常（==0，理论不可达）
// 时按空 tile 域早退，消除 (m+baseM-1)/baseM 与 (n+baseN-1)/baseN 的
// 除零未定义行为。守卫取
// ==0 等值判断形式（CI 静态检查只认 ==0/!=0，<=/>= 不识别）；除零
// UB 仅在除数为 0 时存在，负值 int 除法非 UB 且理论不可达，安全性
// 与 <=0 形式等价。
__aicore__ inline bool SpltFusedEpilogueEarlyExit(int32_t m, int32_t baseM,
                                                  int32_t baseN, int32_t n)
{
    const uint32_t taskRation = GetTaskRation();
    if (taskRation == 0 || baseM == 0 || baseN == 0) { return true; }
    const int64_t totalTiles = static_cast<int64_t>((m + baseM - 1) / baseM) *
                               static_cast<int64_t>((n + baseN - 1) / baseN);
    return totalTiles <= 0;
}

// SpltFusedEpilogueImpl 模板化为 (T, CType, OutType)。AccType 经
// SpltL0CTypeTrait<T> 推导。CType 为 C 矩阵实际 dtype（INT8 A/B 的
// 超集行下可与 T 不同，其余与 T 同型）。非 INT8 路径：CType == OutType
// == T。SpltCdCtx 构造（splt_make_cd_ctx）与 ReLU 掩码缓冲
// （splt_init_relu_mask）为 matmul_kernel_common.h 的共享 helper。
// 早退守卫见 SpltFusedEpilogueEarlyExit。
template <typename T, typename CType = T, typename OutType = T>
__aicore__ inline void SpltFusedEpilogueImpl(GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    using AccType = typename SpltL0CTypeTrait<T>::type;
    using BiasType = SpltBiasDType<T>;
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t alphaVectorScaling = td.alphaVectorScaling;
    const int32_t betaVectorScaling = td.betaVectorScaling;
    const uint64_t biasDevPtr = td.biasDevPtr;
    const int64_t biasStride = td.biasStride;
    const int32_t biasChunkMode = td.biasChunkMode;
    if (SpltFusedEpilogueEarlyExit(m, baseM, baseN, td.n)) { return; }
    GlobalTensor<CType> cGM;
    GlobalTensor<OutType> dGM;
    GlobalTensor<float> alphaVecGM, betaVecGM;
    SpltInitScalingGM(alphaVecGM, betaVecGM, td.alphaDevPtr, td.betaDevPtr,
                       m, alphaVectorScaling, betaVectorScaling);
    GlobalTensor<BiasType> biasVecGM;
    TBuf<TPosition::VECCALC> biasVecBuf;
    TBuf<TPosition::VECCALC> biasRawBuf;
    LocalTensor<float> biasVecUB;
    TPipe pipe;
    const uint64_t maxMAlign = static_cast<uint64_t>((baseM + 1) & ~1);
    const uint64_t maxNAlign = static_cast<uint64_t>((baseN + SPLT_L0C_C0 - 1) / SPLT_L0C_C0) * SPLT_L0C_C0;
    TBuf<TPosition::VECCALC> accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf;
    TBuf<TPosition::VECCALC> alphaVecBuf, betaVecBuf;
    TBuf<TPosition::VECCALC> wideCdBuf;
    SpltCdCtx cd = splt_make_cd_ctx(wideCdBuf, td.ldC, td.ldD, td.cOrderIsCol != 0);
    SpltFusedEpilogueInitBuffers<T, CType, OutType>(pipe, accBuf, dFp32Buf, dTBuf, cTBuf,
        cFp32Buf, tempBuf, halfBuf, wideCdBuf, cd.isCol, maxMAlign, maxNAlign, td.alpha, td.beta,
        alphaVectorScaling, betaVectorScaling,
        biasDevPtr, biasChunkMode, td.activationType);
    // bias UB 的分配 + 加载放在 accBuf 之后（accBuf 必须保持在
    // UB 偏移 0，以匹配 AIC 的 CopyL0C2UB 目标地址）。
    SpltFusedEpilogueInitBias<BiasType>(biasVecGM, biasVecBuf, biasRawBuf, biasVecUB,
                                         pipe, biasDevPtr, biasStride, biasChunkMode,
                                         m, maxMAlign);
    LocalTensor<float> alphaVecUB, betaVecUB;
    // 向量族 chunk 模式（biasChunkMode==1）下按 tile 行数上界分配
    // chunk 槽位（per-tile 惰性加载，见 SpltFusedEpilogueGeneralPath）；
    // 全量模式（biasChunkMode==0）维持 m*4 全量加载不变。
    SpltLoadScalingUB(alphaVecBuf, betaVecBuf, alphaVecUB, betaVecUB,
                       alphaVecGM, betaVecGM, pipe, m,
                       alphaVectorScaling, betaVectorScaling,
                       biasChunkMode, static_cast<uint32_t>(maxMAlign));
    // ReLU b8 位掩码缓冲：行级（maxNAlign 元素 -> <= maxNAlign/8 字节，64B 覆盖）
    TBuf<TPosition::VECCALC> reluMaskBuf;
    LocalTensor<uint8_t> reluMaskUB = splt_init_relu_mask(pipe, reluMaskBuf);
    SpltFusedEpilogueBatchLoop<T, CType, OutType>(td, cd, cGM, dGM, biasVecGM, biasVecUB, biasRawBuf,
        accBuf, dFp32Buf, dTBuf, cTBuf, cFp32Buf, tempBuf, halfBuf,
        alphaVecUB, betaVecUB, alphaVecGM, betaVecGM, reluMaskUB, cGm, dGm);
}

// ============================================================================
// 融合 matmul+epilogue kernel 入口（splitK==1 路径）。
// 每种 dtype 一个入口（BF16 / INT8 / INT8_I32）。
// ============================================================================
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_fp16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<__fp16>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<__fp16>(cGm, dGm, tilingGm);
    }
}

extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_fp32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<float>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<float>(cGm, dGm, tilingGm);
    }
}

// BF16 融合。
// BF16 Cast 使用 Vector Cast<bfloat16_t, float>（CANN 9.1.0
// 的 cast_round_all 已支持）。cube Mmad（tensor_api）将 BF16 走
// NORMAL 路径（与 dtype 无关）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<__bf16>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<__bf16>(cGm, dGm, tilingGm);
    }
}

// INT8 融合 —— INT8 输出（饱和 Cast）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int8_t>(cGm, dGm, tilingGm);
    }
}

// INT8 融合 —— INT32 输出（经 ROUND Cast 的 int32 直接写）。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8_i32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int8_t, int32_t>(cGm, dGm, tilingGm);
    }
}
// INT8 A/B × C/D 超集组合（CType 贯通）：C=D=INT32 / FP16 / BF16。
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8_ci32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, int32_t, int32_t>(cGm, dGm, tilingGm);
    }
}
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8_cf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, __fp16, __fp16>(cGm, dGm, tilingGm);
    }
}
extern "C" __global__ __aicore__ __mix__(1, 2) void splt_fused_matmul_kernel_int8_cbf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR cGm, GM_ADDR dGm, GM_ADDR tilingGm)
{
    if ASCEND_IS_AIC {
        SpltFusedMatmulCubeImpl<int8_t>(aPrunedGm, bGm, dGm, tilingGm);
    } else {
        SpltFusedEpilogueImpl<int8_t, __bf16, __bf16>(cGm, dGm, tilingGm);
    }
}

// 融合 matmul+epilogue 启动器（splitK==1 路径）。
// outDataType —— INT8 路径选择 INT8 或 INT32 输出入口。
extern "C" void splt_fused_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm,
    GM_ADDR cGm, GM_ADDR dGm,
    GM_ADDR tilingGm, int32_t dataType, int32_t cDataType, int32_t outDataType,
    uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_fused_matmul_kernel_fp32<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_fused_matmul_kernel_fp16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_fused_matmul_kernel_bf16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, cGm, dGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            // C/D 超集分派：C=D=INT32 / FP16 / BF16（CType 贯通）；
            // C=INT8 时按 outDataType 分 INT8/INT32（既有）。
            if (cDataType == SPLT_DTYPE_INT32) {
                splt_fused_matmul_kernel_int8_ci32<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            } else if (cDataType == SPLT_DTYPE_FP16) {
                splt_fused_matmul_kernel_int8_cf16<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            } else if (cDataType == SPLT_DTYPE_BF16) {
                splt_fused_matmul_kernel_int8_cbf16<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            } else if (outDataType == SPLT_DTYPE_INT32) {
                splt_fused_matmul_kernel_int8_i32<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            } else {
                // 不可达兜底（防御保留）：host 侧 launch 前置校验
                // （validate_launch_dtypes 配对白名单）已保证到达本分支的
                // INT8 组合为 (C=INT8, D=INT8)——其余 C/D 组合在 launch 前
                // 即被拒绝（NOT_SUPPORTED）。
                splt_fused_matmul_kernel_int8<<<blockDim, nullptr, stream>>>(
                    aPrunedGm, bGm, cGm, dGm, tilingGm);
            }
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in fused matmul launcher\n"); return;
    }
}
