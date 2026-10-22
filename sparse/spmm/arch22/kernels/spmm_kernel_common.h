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

#include "kernel_operator.h"
#include "../spmm.h"

/**
 * @brief 实数版与 complex64 版 SpMM kernel 的公共骨架。
 *
 * 两版 kernel 在三件事上完全一致：读 tiling 里与值类型无关的那部分字段、按 host
 * 端装箱结果确定本核负责的行区间、以及按列 chunk 遍历这些行。真正的差异只在
 * "一个 chunk 内怎么算"（实数是 Axpy 规约，复数是 AoS 双累加器 + J 算子）。
 *
 * 这里把三者收拢到一个非模板基类里，派生类只需提供
 * ProcessChunk(colStart, chunkLen, chunkAlign)。基类刻意不做成模板：实数版
 * 派生类本身是 template<ValT>，若基类依赖模板参数，派生类里所有对继承成员的
 * 非限定引用都会因两阶段名字查找失败，必须逐处加 this->。只把需要回调派生类的
 * 那一个方法做成成员模板，就能避开这个问题。
 *
 * alpha/beta 不放在这里：实数版是一个 float，复数版是 (re, im) 两个。
 */
class SpmmArch22KernelCommon {
protected:
    // tiling 里与值类型无关的字段。K 不在此处：实数版存成员，复数版直接用 td->K
    // 算 GlobalTensor 元素数，没有成员。
    __aicore__ inline void InitCommonTiling(__gm__ SpmmArch22TilingData *td)
    {
        M = td->M;
        N = td->N;
        nnz = td->nnz;
        nChunks = td->nChunks;
        // skipBeta 不在这里定：它取决于 beta，而 beta 实数版是一个 float、复数版
        // 是 (re, im) 两个，都在派生类的 InitTiling 里读。派生类必须设置它。
        //
        // 语义以 torch.sparse.addmm 为准：beta==0 时它【完全忽略 C】(BLAS 惯例)，
        // 不做 0*C 的 IEEE 运算，所以 C 里的 NaN/Inf 不传播。此前这里硬编码
        // skipBeta=false，依据是"torch 算的是 0.0*NaN=NaN"——该前提经最小用例与
        // 官方 252 例 beta=0 用例的 golden 重建双向核对，确认【不成立】。
        // idxBase=1(Fortran 兼容)时 rowOffsets/colIndices 是 1-based，减去
        // idxBaseOffset 换算成 0-based 数组下标；idxBase=0 时此值恒为 0。
        idxBaseOffset = td->idxBaseHost;
        // host 实际用的装箱粒度。分组路径只有在它与 rowGroup 相等时才能断定
        // 一组内的行号连续，见 ProcessChunkGrouped 里的 blockPacked。
        rowBlockHost = static_cast<uint32_t>(td->rowBlockHost > 0 ? td->rowBlockHost : 0);
        // 分组路径走流水实现还是原实现。由 host 决定并写进 tiling，kernel 不自行
        // 推断：两侧的槽位数/分组行数必须取自同一组预算函数，见 spmm.h 的
        // SpmmArch22PipeRowGroup。
        pipeMode = (td->pipeModeHost != 0);
        // 流水路径是否把 B 行偏移向量化预算（见 spmm_kernel_group.h 的 kVecRowOffset）。
        // host 侧已保证仅在 int32 不溢出且 nnz 够大时置 1，小矩阵回落标量路径不退步。
        vecRowOffset = (td->vecRowOffsetHost != 0);
    }

    // 本核负责 reorder 序列里 [myRowStart, myRowEnd) 这一段行，边界由 host 侧
    // 按行度数均衡后写在 workspace 的 binEdge 数组里。
    __aicore__ inline void InitRowRange(__gm__ SpmmArch22TilingData *td, GM_ADDR workspace)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        __gm__ int32_t *binEdge =
            reinterpret_cast<__gm__ int32_t *>(workspace + td->binEdgeOffset);
        myRowStart = static_cast<uint32_t>(binEdge[blockId]);
        myRowEnd = static_cast<uint32_t>(binEdge[blockId + 1]);
        reorderGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(workspace + td->reorderOffset), M);
    }

    // 取回一组的行号与每行的非零区间，填进 grpRowIds/grpRs/grpRe，返回组内行号
    // 是否连续。两版 kernel 的行元数据布局完全一样（都是 reorder + rowOffsets 的
    // int32 数组），这一步没有值类型的参与，所以放在基类里共用；GroupCtx 的其余
    // 字段各版形状不同，仍由派生类自己填。
    //
    // host 按 rowBlockHost 行一块做连续行块装箱时，组边界与块边界重合(两边的粒度
    // 都取自 SpmmArch22RowGroup)，一组就是一个整块、行号连续，于是能用一次连续
    // DMA 取回整组的 rowOffsets，省掉 gCnt 次 reorder 读和 2*gCnt 次散落的
    // rowOffsets 读——探针测出每行残留开销 754ns，主要就是这些散读加 C 的两次 DMA。
    //
    // blockPacked 这个前提不能省。只查首尾两项是否相差 gCnt-1 会误判：逐行装箱下
    // reorder 按行度数排序，[5, 9, 3, 8] 这种首尾恰好对上、中间乱序的排列很常见
    // （同度数行一多就撞），据此按连续区间去取 rowOffsets 会算出错误结果。
    // host 走逐行装箱时一律退回逐行读。
    __aicore__ inline bool LoadGroupRows(uint32_t gBase, uint32_t gCnt, int32_t &row0,
        AscendC::DataCopyPadParams padParams)
    {
        const bool blockPacked = (rowBlockHost == rowGroup);
        row0 = reorderGm.GetValue(gBase);
        const bool contig = blockPacked &&
            (reorderGm.GetValue(gBase + gCnt - 1) == row0 + static_cast<int32_t>(gCnt) - 1);
        if (contig) {
            AscendC::LocalTensor<int32_t> offTile = rowOffBuf.Get<int32_t>();
            AscendC::DataCopyParams offCopy{
                1, static_cast<uint16_t>((gCnt + 1) * sizeof(int32_t)), 0, 0};
            AscendC::DataCopyPad(offTile, rowOffsetsGm[row0], offCopy, padParams);
            auto evOff = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evOff);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evOff);
            for (uint32_t g = 0; g < gCnt; g++) {
                grpRowIds[g] = static_cast<uint32_t>(row0) + g;
                grpRs[g] = static_cast<uint32_t>(offTile.GetValue(g) - idxBaseOffset);
                grpRe[g] = static_cast<uint32_t>(offTile.GetValue(g + 1) - idxBaseOffset);
            }
            return true;
        }
        for (uint32_t g = 0; g < gCnt; g++) {
            int32_t row = reorderGm.GetValue(gBase + g);
            grpRowIds[g] = static_cast<uint32_t>(row);
            grpRs[g] = static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseOffset);
            grpRe[g] = static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseOffset);
        }
        return false;
    }

    // 把组内游标 (g, idx) 推到下一个真实存在的非零元上，跳过空行。返回 false
    // 表示这一组已经走完。填批槽和累加时各走一遍同样的推进，靠它保持一致。
    __aicore__ inline bool AdvanceGroupCursor(const uint32_t *rsArr, const uint32_t *reArr,
        uint32_t gCnt, uint32_t &g, uint32_t &idx) const
    {
        while (g < gCnt && idx >= reArr[g]) {
            g++;
            if (g < gCnt) {
                idx = rsArr[g];
            }
        }
        return g < gCnt;
    }

    // 列分块遍历。self 传派生类 this，回调其 ProcessChunk；派生类需把本类声明为
    // friend（ProcessChunk 保持 private）。
    template <typename Derived>
    __aicore__ inline void ProcessChunks(Derived *self)
    {
        if (myRowStart >= myRowEnd) { return; }

        for (uint32_t chunkIdx = 0; chunkIdx < nChunks; chunkIdx++) {
            uint32_t colStart = chunkIdx * nMaxPerChunk;
            uint32_t colEnd = colStart + nMaxPerChunk;
            if (colEnd > N) { colEnd = N; }
            uint32_t chunkLen = colEnd - colStart;
            uint32_t chunkAlign = ROUNDUP(chunkLen, SPMM_ARCH22_ALIGN);

            self->ProcessChunk(colStart, chunkLen, chunkAlign);
        }
    }

    AscendC::GlobalTensor<int32_t> reorderGm;
    AscendC::GlobalTensor<int32_t> rowOffsetsGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowOffBuf;  // 连续行块整组的 rowOffsets

    uint32_t M, N, nnz;
    uint32_t nChunks, nMaxPerChunk, nAlignChunk;
    uint32_t myRowStart, myRowEnd;
    bool skipBeta;
    int32_t idxBaseOffset;
    uint32_t rowBlockHost;
    uint32_t batchSlots;  // 一批 B 行的槽位数（流水模式下是每个半区的槽位数），0 表示 UB 装不下
    uint32_t rowGroup;    // ProcessChunkGrouped 一组的行数，0 表示退回逐行路径
    bool fuseGroupDma;    // 整组 C 能否合成一次多块 DMA，见各版 InitLocalBuffers
    bool pipeMode;        // 分组路径是否走软件流水实现，取自 tiling
    bool vecRowOffset;    // 流水路径是否向量化预算 B 行偏移，取自 tiling

    // 当前组的行号与各行非零区间，由 LoadGroupRows 填。放成员而不是局部数组，
    // 是为了让分组路径的几个步骤共享，不必逐层传数组指针。
    uint32_t grpRowIds[SPMM_ARCH22_ROW_GROUP_MAX];
    uint32_t grpRs[SPMM_ARCH22_ROW_GROUP_MAX];
    uint32_t grpRe[SPMM_ARCH22_ROW_GROUP_MAX];
};
