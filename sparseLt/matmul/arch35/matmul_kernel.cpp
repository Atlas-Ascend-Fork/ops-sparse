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
 * \file matmul_kernel.cpp
 * \brief sparseLt matmul 非融合 cube kernel（splitK>1 路径）+ extern "C" 启动器。
 * 超大源文件整改：原单文件（nbnc 2147 > 2000）按功能族拆分——
 *   - matmul_kernel.cpp          本文件：非融合 cube matmul（AIC_ONLY）
 *   - matmul_epilogue_kernel.cpp 非融合 epilogue（AIV_ONLY）
 *   - matmul_fused_kernel.cpp    融合 matmul+epilogue（__mix__(1,2)，splitK==1）
 * 跨文件共享的设备侧基础设施位于 matmul_kernel_common.h；各文件内容与
 * 拆分前逐字一致（结构性重构，零行为变更）。
 * kernel：
 *   - matmul_kernel (AIC_ONLY, Te::Mmad tensor_api): M(m,k) x N(k,n) -> temp (splitK*m*n)
 * A/B 的 GM 布局经 NDExt/DNExt 的行距参数（td.ldA/td.ldB）直读物理 ld，
 * 行尾 padding 由 MTE2 搬运逐行跳过，dense 侧带 ld 的输入无需预处理。
 * 启动器在 ASC 编译单元内封装 <<<>>>；host.cpp 以普通 C 函数形式调用。
 * 与 prune_kernel（tiling 经启动参数块按值传入）不同，
 * matmul/epilogue/fused 各 kernel 通过 splt_load_tiling(tilingGm) 从 GM 加载 tiling。
 * TilingData 结构体与 splt_load_tiling 定义在 shared/aclsparselt_internal.h。
 */


#include <cstdint>

#include "log/log.h"

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "shared/aclsparselt_internal.h"
#include "matmul_kernel.h"
#include "matmul_kernel_common.h"

// ============================================================================
// SpltMatmulCubeImpl 的 tile 循环驱动：m/n tile 遍历 + GM 切片计算 +
// K 循环 + Fixpipe L0C->GM。flag 操作保持不变。
// ============================================================================
template <typename T, typename GmTensorA, typename GmTensorB>
__aicore__ inline void SpltMatmulCubeTileLoop(
    const AclsparseltTilingData& td,
    GmTensorA& gmA, GmTensorB& gmB, GM_ADDR tempGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalMNTiles, int64_t totalTiles)
{
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t baseK = td.baseK;
    const int32_t kSegLen = td.kSegLen;

    for (int32_t tileIdx = blockId; tileIdx < totalTiles; tileIdx += blockNum) {
        const int32_t kSeg = static_cast<int32_t>(static_cast<int64_t>(tileIdx) / totalMNTiles);
        const int32_t mnTile = static_cast<int32_t>(static_cast<int64_t>(tileIdx) % totalMNTiles);
        const int32_t mTile = mnTile / nTiles;
        const int32_t nTile = mnTile % nTiles;

        const int32_t mPos = mTile * baseM;
        const int32_t nPos = nTile * baseN;
        const int32_t kSegStart = kSeg * kSegLen;
        const int32_t curM = (mPos + baseM <= m) ? baseM : (m - mPos);
        const int32_t curN = (nPos + baseN <= n) ? baseN : (n - nPos);

        // 跳过起始位置超出实际 k 的 kSeg tile。
        if (kSegStart >= k) {
            continue;
        }

        // 两级 Slice，与 cann-samples matmul_kernel_swat.h 一致。
        auto gmBlockARow = gmA.Slice(Te::MakeCoord(mPos, 0), Te::MakeShape(curM, k));
        // B 切片对 transB 与非 transB 相同：transB 时 GM B 声明为
        // DNExt(k,n)（(n,k) 物理数据的列主序视图），因此按 (0, nPos, k, curN)
        // 切片对两条路径均成立。
        auto gmBlockBCol = gmB.Slice(Te::MakeCoord(0, nPos), Te::MakeShape(k, curN));

        // 本 kSeg 的 GM 输出切片：tempSeg 位于 [kSeg*m*n]。
        // temp 元素类型跟随 L0C 类型（FP32/FP16/BF16 为 float，
        // INT8 为 int32_t）。sizeof(float) == sizeof(int32_t) == 4，
        // 字节偏移计算完全一致；仅类型化指针不同。
        using L0CType = typename SpltL0CTypeTrait<T>::type;
        __gm__ uint8_t* tempBase = reinterpret_cast<__gm__ uint8_t*>(tempGm) +
                                   static_cast<int64_t>(kSeg) * m * n * sizeof(L0CType);
        auto gmCSeg = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ L0CType*>(tempBase)),
                                 Te::MakeFrameLayout<Te::NDExtLayoutPtn, SPLT_GM_FLOAT_C0>(m, n));

        auto tensorL0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, typename SpltL0CTypeTrait<T>::type>(0),
                                     Te::MakeFrameLayout<Te::NZLayoutPtn, SPLT_L0C_C0>(curM, curN));

        // 写 L0C 前等待前一个 tile 的 Fixpipe 完成。
        WaitFlag<HardEvent::FIX_M>(0);
        bool cmatrixInitVal = true;
        SpltKL1Loop<T>(gmBlockARow, gmBlockBCol, tensorL0C,
                       curM, curN, k, kSegLen, kSegStart, baseK, td.kL1Size,
                       cmatrixInitVal, l1cfg);

        // ---- L0C -> GM (Fixpipe).
        auto gmBlockC = gmCSeg.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(curM, curN));
        Te::FixpipeParams fixpParams;
        fixpParams.unitFlag = static_cast<uint8_t>(SPLT_FINAL_ACCUMULATION);
        auto copyL0C2GM = Te::MakeCopy(Te::CopyL0C2GM{});
        Te::Copy(copyL0C2GM.with(fixpParams), gmBlockC, tensorL0C);
        SetFlag<HardEvent::FIX_M>(0);
    }
}

// SpltMatmulCubeImpl 的 transB 分发：构造 gmB（依据 transB 选择
// DNExt 或 NDExt）并调用 SpltMatmulCubeTileLoop。消除重复的
// transB 分支代码（4 个分支 -> 2 个 + 辅助函数）。
// B 的物理行距经 td.ldB 直通布局参数：dense B 从描述符 ld 直读（ldB==k/n
// 的紧凑场景与旧逻辑维度参数逐位一致——零回归锚点）；B_pruned（B-sparse）
// 恒为 prune 紧凑输出（ldB==n）。MTE2 搬运按行距逐行跳过行尾 padding，
// 非紧凑 dense B 无需任何预处理。
template <typename T, typename GmTensorA>
__aicore__ inline void SpltMatmulCubeDispatchB(
    const AclsparseltTilingData& td, GmTensorA& gmA, GM_ADDR bGm, GM_ADDR tempGm,
    const L1BufferConfig& l1cfg, int32_t blockId, int32_t blockNum,
    int32_t nTiles, int64_t totalMNTiles, int64_t totalTiles)
{
    const int32_t k = td.k;
    const int32_t n = td.n;
    if (td.transB != 0) {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(td.ldB, n));
        SpltMatmulCubeTileLoop<T>(td, gmA, gmB, tempGm, l1cfg, blockId, blockNum,
                                   nTiles, totalMNTiles, totalTiles);
    } else {
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ T*>(bGm)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(k, td.ldB));
        SpltMatmulCubeTileLoop<T>(td, gmA, gmB, tempGm, l1cfg, blockId, blockNum,
                                   nTiles, totalMNTiles, totalTiles);
    }
}

// ============================================================================
// CUBE MATMUL KERNEL（AIC_ONLY，Te::Mmad tensor_api）
//   M(m,k) x N(k,n) -> temp(splitK*m*n, FP32)
//   每 tile 单缓冲，K 循环在 L0C 中累加。正确性优先。
// ============================================================================
// GM 输入张量布局使用 Te::LayoutTraitDefault<T>，其 C0 由数据类型推导
// （32/sizeof(T)：fp16=16，fp32=8），因此无需显式 C0 参数。
template <typename T>
__aicore__ inline void SpltMatmulCubeImpl(GM_ADDR aPrunedGm, GM_ADDR bGm,
                                          GM_ADDR tempGm, GM_ADDR tilingGm)
{
    const AclsparseltTilingData td = splt_load_tiling(tilingGm);
    const int32_t m = td.m;
    const int32_t n = td.n;
    const int32_t k = td.k;
    const int32_t baseM = td.baseM;
    const int32_t baseN = td.baseN;
    const int32_t splitK = td.splitK;
    // batch 参数
    const int32_t numBatches = (td.numBatches > 0) ? td.numBatches : 1;

    const int32_t blockId = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());

    const int32_t mTiles = (m + baseM - 1) / baseM;
    const int32_t nTiles = (n + baseN - 1) / baseN;
    // 使用 int64_t 防止 mTiles*nTiles 接近
    // INT32_MAX（大 m、n）时溢出。
    const int64_t totalMNTiles = static_cast<int64_t>(mTiles) * static_cast<int64_t>(nTiles);
    // 每个 batch 内的 totalTiles（batch 循环在外层）
    const int64_t totalTiles = totalMNTiles * static_cast<int64_t>(splitK);
    if (totalTiles <= 0) { return; }

    // GM 基础张量（ND 行主序入口）。
    // GM 侧 Te::NDExtLayoutPtn 必须使用与 dtype 匹配的 C0
    // （32/sizeof(T)：fp16=16，fp32=8）。默认 Te::LayoutTraitDefault<> 以
    // uint16_t 推导 => C0=16，对 fp32 是错误的，会导致 CopyGmToCbufMultiND2Nz
    // 按错误的内块大小打包 NZ 分形，数据错乱。
    // 使用 Te::LayoutTraitDefault<T>，它同时设置数据类型与 C0（=32/sizeof(T)）。
    // A 的物理行距经 td.ldA 直通布局参数（(b) 两侧恒描述符 ld 快照——
    // A_pruned（A-sparse）为 prune 的描述符 (order, ld) 布局输出，dense A
    //（B-sparse / dense×dense）从描述符 ld 直读；紧凑场景与旧逻辑维度
    // 参数逐位一致——零回归锚点），MTE2 按行距跳过行尾 padding，无需预处理。
    // sparseTrans=1：声明为 DNExt(ldA, k)（列主序视图），使 CopyGM2L1
    // 在分形转换时完成转置 —— 与 transB 同一机制（基值含 op 分量，
    // COL 分量由执行期 splt_apply_gm_layout 翻转叠加，最终标志 =
    // isColView(该侧)，投递到 L0 的逻辑矩阵恒为 op(该侧)）。
    // sparseTrans=0：标准 NDExt(m, ldA)（行主序视图）。
    // transB=1：声明为 DNExt(ldB, n)（列主序）—— CopyGM2L1 自动选择
    // DN2ZN 完成转置。
    // DNExt 与 NDExt 产生不同的张量类型，因此 tile 循环调用必须在
    // 调用点分支（按 GmTensorA/GmTensorB 类型做模板分发）。
    const L1BufferConfig l1cfg = InitL1DoubleBuffer<T>(baseM, baseN, td.kL1Size);
    SetMMLayoutTransform(true);
    // batch 外循环（GM 指针按 batchStride 偏移）
    using L0CType = typename SpltL0CTypeTrait<T>::type;
    const int64_t tempBatchStride = static_cast<int64_t>(splitK) * m * n * sizeof(L0CType);
    for (int32_t b = 0; b < numBatches; ++b) {
        // GM 指针按 batchStride 偏移（batchStride 以元素数为单位）
        __gm__ T* aBase = reinterpret_cast<__gm__ T*>(aPrunedGm)
                          + static_cast<int64_t>(b) * td.batchStrideA;
        GM_ADDR bBatchGm = bGm + static_cast<int64_t>(b) * td.batchStrideB * static_cast<int64_t>(sizeof(T));
        GM_ADDR tempBatchGm = tempGm + static_cast<int64_t>(b) * tempBatchStride;
        if (td.sparseTrans != 0) {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::DNExtLayoutPtn, Te::LayoutTraitDefault<T>>(td.ldA, k));
            SpltMatmulCubeDispatchB<T>(td, gmA, bBatchGm, tempBatchGm, l1cfg, blockId, blockNum,
                                        nTiles, totalMNTiles, totalTiles);
        } else {
            auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aBase),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<T>>(m, td.ldA));
            SpltMatmulCubeDispatchB<T>(td, gmA, bBatchGm, tempBatchGm, l1cfg, blockId, blockNum,
                                        nTiles, totalMNTiles, totalTiles);
        }
    }

    // 排空所有残留 flag（L1 两个乒乓槽位 + L0 + L0C）。
    DrainCubeFlags();

    // 恢复默认布局转换模式（与 cann-samples
    // matmul_kernel_swat.h:162 一致）。
    SetMMLayoutTransform(false);
}

// ============================================================================
// kernel 入口（每种 dtype 一个），由 host 启动器分发。
// 每种 dtype 一个入口。INT8 matmul（cube）为 INT8->INT8 与
// INT8->INT32 共享（cube 只写 int32 temp；输出类型分发
// 在 epilogue/fused 中完成）。
// ============================================================================

// --- matmul (cube) ---
extern "C" __global__ __aicore__ void splt_matmul_kernel_fp16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<__fp16>(aPrunedGm, bGm, tempGm, tilingGm);
}
extern "C" __global__ __aicore__ void splt_matmul_kernel_fp32(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<float>(aPrunedGm, bGm, tempGm, tilingGm);
}
// BF16 cube。
// cube Mmad（tensor_api）将 BF16 走 NORMAL 路径（与 dtype 无关）；
// cube-only kernel 不涉及 Vector Cast。L0C=float 由 SpltL0CTypeTrait 推导。
extern "C" __global__ __aicore__ void splt_matmul_kernel_bf16(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<__bf16>(aPrunedGm, bGm, tempGm, tilingGm);
}
// INT8 cube（经 SpltL0CTypeTrait 得 L0C=int32_t；INT8->INT8 / INT8->INT32 共享）。
extern "C" __global__ __aicore__ void splt_matmul_kernel_int8(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SpltMatmulCubeImpl<int8_t>(aPrunedGm, bGm, tempGm, tilingGm);
}

// ============================================================================
// host 侧启动器（ASC 编译单元）：封装 <<<>>>。host.cpp 以 C 函数形式调用。
// switch 四分支分发（FP32/FP16/BF16/INT8）+ epilogue/fused 的
// INT8 outDataType 子分支（INT8 与 INT32 输出）。
// ============================================================================
extern "C" void splt_matmul_kernel_launch(
    GM_ADDR aPrunedGm, GM_ADDR bGm, GM_ADDR tempGm,
    GM_ADDR tilingGm, int32_t dataType, uint32_t blockDim, void *stream)
{
    switch (dataType) {
        case SPLT_DTYPE_FP32:
            splt_matmul_kernel_fp32<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_FP16:
            splt_matmul_kernel_fp16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_BF16:
            splt_matmul_kernel_bf16<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        case SPLT_DTYPE_INT8:
            splt_matmul_kernel_int8<<<blockDim, nullptr, stream>>>(
                aPrunedGm, bGm, tempGm, tilingGm);
            break;
        default:
            OP_LOGE(kSparseLtLogTag, "unreachable: unknown dtype in matmul launcher\n"); return;
    }
}
