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
#include "../spmm.h"
#include "spmm_kernel_common.h"

// 与 spmm_kernel.h 同理，本文件用相对路径引用上一层的 spmm.h，必须留在 kernels/ 下。

// AscendC 具体名字引入集中在共享头（见其内说明）。本文件在 spmm_kernel.h 的相应
// #include 之前即被包含，故这里显式引入一次（该头有 #pragma once，重复包含无副作用）。
#include "spmm_kernel_using.h"
#include "spmm_kernel_group_axpy.h"

/**
 * @brief 实数版 SpMM 的跨行分组路径，按值类型模板化。
 *
 * 从 spmm_kernel.h 拆出来的一整条执行路径：把一组 gCnt 行的固定开销压成每组一次，
 * 是当前小 N 场景的主力路径（逐行路径只在 UB 放不下两行累加器时兜底）。拆分的直接
 * 动因是把 spmm_kernel.h 的行数压回门禁的头文件上限以内，同时这条路径与逐行路径
 * 之间本来只通过 ProcessChunk 的一次分发相接，切口很干净。
 *
 * 值类型相关的 GM 句柄、alpha/beta、以及分组与整批路径共用的 UB 缓冲都落在这一层，
 * 由派生类 SpmmArch22Kernel 在 InitGlobalBuffers/InitLocalBuffers 里填。派生类是
 * template<ValT>，基类也依赖 ValT，因此派生类里对继承成员的非限定引用会因两阶段
 * 名字查找失败——那边用一组 using 声明把这些名字重新引入作用域，而不是逐处加
 * this->，见 spmm_kernel.h 里 kUsingBase 附近的说明。
 *
 * @tparam ValT 非零值 / B / C 的存储类型（三者当前要求一致）。
 */
template <typename ValT>
class SpmmArch22GroupPath : public SpmmArch22KernelCommon {
protected:
    static constexpr bool kIsFloat = std::is_same_v<ValT, float>;

    // bf16 走整批搬运路径，float/half 走逐元素标量路径。根因是 bf16 的非零值读不了
    // 标量（后端不支持 bfloat16_t 的 static_cast<float>），只能整批 DMA 进 UB 再向量
    // Cast。
    //
    // 两条路径都实测过：整批搬运把 bf16 的 P-02(N=128) 从 9680us 降到 5937us
    // (1.63x)，因为它消掉了那条 bf16 专用的逐元素 DMA+Cast 链；但对 float/half 反而
    // 更慢(fp32 的 P-02 5415->6120us，N=1433 的 P-01 67->86us)。原因是这两个 dtype
    // 的标量取值本来就只是一条 GetValue，整批搬运替掉它要付两次 tile DMA 加两组跨
    // 引擎同步事件，而稀疏矩阵的行度数通常只有个位数，摊不平这笔开销。小 N 下真正的
    // 瓶颈是 B 行 gather 的 DMA 带宽(P-02 fp32 实测约 110GB/s，MTE2 占比 0.775)，
    // 不是标量取值延迟。
    static constexpr bool kUseNzTile = std::is_same_v<ValT, bfloat16_t>;

    // 一组行的共享标量。分组路径的四个步骤都要用这批值，逐个传参会让每个函数
    // 都挂七八个形参，聚成一个结构体按引用传。
    struct GroupCtx {
        uint32_t gCnt;       // 本组实际行数，尾组可能不足 rowGroup
        uint32_t span;       // gCnt * nAlignChunk，本组在累加器里占的元素数
        uint32_t colStart;
        uint32_t chunkLen;
        int32_t row0;        // 组内首行的行号
        bool contig;         // 组内行号是否连续
        bool fused;          // 整组的 C 能否合成一条多块 DMA
        uint64_t groupBase;  // contig 时整组 C 在 GM 里的起始偏移
    };

    // 累加器初值。beta!=0 时直接把 C 读进来乘 beta，省掉逐行路径里"先清零、
    // 再 Muls 到 tmp、再 Add"的那一轮：一组只有一次 Muls，且不需要 tmpBuf。
    //
    // 前提：本函数写入的 acc/stage 必须是按 rowGroup 整组宽度独立分配的缓冲，
    // 不得复用 B 行的批槽。ROW_GROUP_MAX(64) 比 BATCH_SLOTS_MAX(32) 大一倍，
    // 借批槽会在 gc.gCnt > slots 时越界写进相邻缓冲——complex64 路径曾因此
    // 让未乘 beta 的 C 污染累加器实部。改这里时不要引入批槽复用。
    __aicore__ inline void InitGroupAcc(LocalTensor<float> &acc, LocalTensor<ValT> &stage,
        const GroupCtx &gc, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        if (skipBeta) {
            Duplicate(acc, 0.0f, gc.span);
            return;
        }
        if (gc.fused) {
            DataCopyParams groupCopy{static_cast<uint16_t>(gc.gCnt),
                static_cast<uint16_t>(gc.chunkLen * sizeof(ValT)), 0, 0};
            if constexpr (kIsFloat) {
                DataCopyPad(acc, cGm[gc.groupBase], groupCopy, padParams);
            } else {
                DataCopyPad(stage, cGm[gc.groupBase], groupCopy, padParams);
            }
        } else {
            for (uint32_t g = 0; g < gc.gCnt; g++) {
                const uint64_t outBase =
                    static_cast<uint64_t>(grpRowIds[g]) * N + gc.colStart;
                if constexpr (kIsFloat) {
                    DataCopyPad(acc[g * nAlignChunk], cGm[outBase], copyParams, padParams);
                } else {
                    DataCopyPad(stage[g * nAlignChunk], cGm[outBase], copyParams, padParams);
                }
            }
        }
        auto evC = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evC);
        WaitFlag<HardEvent::MTE2_V>(evC);
        if constexpr (!kIsFloat) {
            Cast<float, ValT>(acc, stage, RoundMode::CAST_NONE, gc.span);
            PipeBarrier<PIPE_V>();
        }
        if (betaVal != 1.0f) {
            Muls(acc, acc, betaVal, gc.span);
        }
    }

    // 写回。fp32 直接从累加器搬出；窄类型先整组 Cast 到暂存再搬。
    __aicore__ inline void StoreGroupAcc(LocalTensor<float> &acc, LocalTensor<ValT> &stage,
        const GroupCtx &gc, DataCopyParams copyParams)
    {
        if constexpr (!kIsFloat) {
            Cast<ValT, float>(stage, acc, RoundMode::CAST_ROUND, gc.span);
        }
        auto evOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evOut);
        WaitFlag<HardEvent::V_MTE3>(evOut);
        if (gc.fused) {
            DataCopyParams groupCopy{static_cast<uint16_t>(gc.gCnt),
                static_cast<uint16_t>(gc.chunkLen * sizeof(ValT)), 0, 0};
            if constexpr (kIsFloat) {
                DataCopyPad(yGm[gc.groupBase], acc, groupCopy);
            } else {
                DataCopyPad(yGm[gc.groupBase], stage, groupCopy);
            }
            return;
        }
        for (uint32_t g = 0; g < gc.gCnt; g++) {
            const uint64_t outBase = static_cast<uint64_t>(grpRowIds[g]) * N + gc.colStart;
            if constexpr (kIsFloat) {
                DataCopyPad(yGm[outBase], acc[g * nAlignChunk], copyParams);
            } else {
                DataCopyPad(yGm[outBase], stage[g * nAlignChunk], copyParams);
            }
        }
    }

    // WAR：下一组会重写 acc/stage——beta 路径由 MTE2 写，skipBeta 路径由
    // Duplicate(PIPE_V) 写，窄类型的 Cast 也在 PIPE_V 上——所以两个方向都要挡。
    // 逐行路径靠 outQueue 的 EnQue/DeQue 隐式兜住这件事，裸 TBuf 没有依赖跟踪。
    __aicore__ inline void GroupWarSync()
    {
        auto evW1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evW1);
        WaitFlag<HardEvent::MTE3_V>(evW1);
        if (!skipBeta) {
            auto evW2 =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(evW2);
            WaitFlag<HardEvent::MTE3_MTE2>(evW2);
        }
    }

    // 处理一个列分块内本核负责的所有行。fp32 路径直接把 outQueue 的 tile 当
    // 累加器用(与基线一致，省一块 UB)，窄类型则用独立的 fp32 accBuf 累加。
    // 把一组 gCnt 行的固定开销压成每组一次。逐行路径每行要付 3 次 GM 标量读、
    // 2 次 GM DMA、2~3 组 TQue 同步；这里换成每组一次 Duplicate/Muls、一组 C 读入
    // 事件、一组写回事件，非零元的整批 gather 还能跨行凑满槽位。
    // 取回一组的行元数据并算好这一组在 UB / GM 里的几何量。
    __aicore__ inline void FillGroupCtx(GroupCtx &gc, uint32_t gBase, uint32_t colStart,
        uint32_t chunkLen, DataCopyPadParams padParams)
    {
        gc.gCnt = myRowEnd - gBase;
        if (gc.gCnt > rowGroup) {
            gc.gCnt = rowGroup;
        }
        gc.colStart = colStart;
        gc.chunkLen = chunkLen;
        gc.contig = LoadGroupRows(gBase, gc.gCnt, gc.row0, padParams);
        gc.span = gc.gCnt * nAlignChunk;
        // 行号连续且 UB 槽位无补白时，整组的 C 在 GM 里也是一整片连续区域，
        // 读入和写出各只要一条多块 DMA(块间步长为零)。
        gc.fused = gc.contig && fuseGroupDma;
        gc.groupBase = static_cast<uint64_t>(gc.row0) * N + colStart;
    }

    __aicore__ inline void ProcessChunkGrouped(
        uint32_t colStart, uint32_t chunkLen, uint32_t chunkAlign)
    {
        DataCopyParams copyParams = {1, (uint16_t)(chunkLen * sizeof(ValT)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        LocalTensor<float> acc = accGroupBuf.Get<float>();
        LocalTensor<ValT> stage;
        if constexpr (!kIsFloat) {
            stage = stageGroupBuf.Get<ValT>();
        }
        LocalTensor<ValT> batch = bBatchBuf.Get<ValT>();
        LocalTensor<float> castBatch;
        if constexpr (!kIsFloat) {
            castBatch = bCastBuf.Get<float>();
        }
        // 流水路径的句柄每个列分块建一次。原路径不用它，构造本身也没有副作用，
        // 但仍按 pipeMode 跳过——非流水模式下 nzIdxBuf 等缓冲可能没有分配。
        PipeCtx pc;
        if (pipeMode) {
            BuildPipeCtx(pc, batch, castBatch, colStart, chunkAlign, copyParams, padParams);
        }

        for (uint32_t gBase = myRowStart; gBase < myRowEnd; gBase += rowGroup) {
            GroupCtx gc;
            FillGroupCtx(gc, gBase, colStart, chunkLen, padParams);

            InitGroupAcc(acc, stage, gc, copyParams, padParams);
            PipeBarrier<PIPE_V>();

            if (pipeMode) {
                AccumulateGroupPipedDispatch(acc, pc, gc);
            } else if (kUseNzTile && !gc.contig) {
                // bf16 的整批值 DMA 要求批内下标连续，跨行只在行号连续时成立。
                // 尾块不满或 host 走了逐行装箱时按每组一行退化——行内下标必然连续。
                // C 的读入/写出和 Duplicate/Muls 仍按整组走，只有累加退化。
                for (uint32_t g = 0; g < gc.gCnt; g++) {
                    LocalTensor<float> accRow = acc[g * nAlignChunk];
                    AccumulateGroup(accRow, batch, castBatch, grpRs + g, grpRe + g, 1, colStart,
                        chunkAlign, copyParams, padParams);
                }
            } else {
                AccumulateGroup(acc, batch, castBatch, grpRs, grpRe, gc.gCnt, colStart,
                    chunkAlign, copyParams, padParams);
            }

            StoreGroupAcc(acc, stage, gc, copyParams);
            if (gBase + rowGroup < myRowEnd) {
                GroupWarSync();
            }
        }
    }

    // 一组内所有行的非零元，按 batchSlots 个一批连续处理，批可以跨行边界。
    //
    // 槽位到累加器的对应关系不另存数组：填槽和累加走同一段确定性的游标推进，
    // 累加时用 (g0, idx0) 重放一遍即可。省下 slotG/slotVal 两个标量数组，
    // 代价只是每个非零元多几条标量运算。
    //
    // 填一批 B 行槽位：从 (g, idx) 起沿组内各行推进，最多填 batchSlots 个。返回
    // 实际填入个数，并把游标推进到下一批起点；firstIdx 是这一批第一个非零元在
    // CSR 里的下标，bf16 的整批值 DMA 要用它做起点。
    __aicore__ inline uint32_t FillGroupBatch(LocalTensor<ValT> &batch, const uint32_t *rsArr,
        const uint32_t *reArr, uint32_t gCnt, uint32_t &g, uint32_t &idx, uint32_t &firstIdx,
        uint32_t colStart, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t cnt = 0;
        while (cnt < batchSlots && AdvanceGroupCursor(rsArr, reArr, gCnt, g, idx)) {
            if (cnt == 0) {
                firstIdx = idx;
            }
            const uint32_t col =
                static_cast<uint32_t>(colIndicesGm.GetValue(idx) - idxBaseOffset);
            DataCopyPad(batch[cnt * nAlignChunk],
                bGm[static_cast<uint64_t>(col) * N + colStart], copyParams, padParams);
            cnt++;
            idx++;
        }
        return cnt;
    }

    // 等这一批的 B 行(以及 bf16 的非零值)搬完，窄类型再整批 Cast 成 fp32。分组
    // 路径与逐行整批路径共用：两边的槽位布局与事件序列完全一致。
    //
    // 窄类型这里一条 Cast 覆盖整批 cnt*nAlignChunk 个元素，而不是每个非零元一次。
    // 这是 P-03 窄类型的主要瓶颈所在：fp32 每个非零元只有一条 Axpy，窄类型原来是
    // Cast + PipeBarrier + Muls + Add 四条。P-03 fp16 只搬 31.7GB 却要 127.7ms
    // (248GB/s，约 HBM 峰值三成)，而 fp32 搬 63.3GB 用 109.9ms(600GB/s，七成半)
    // ——窄类型明显卡在向量侧而不是带宽上。
    __aicore__ inline void SyncAndCastBatch(LocalTensor<ValT> &batch,
        LocalTensor<float> &castBatch, uint32_t cnt)
    {
        auto evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evIn);
        WaitFlag<HardEvent::MTE2_V>(evIn);
        if constexpr (!kIsFloat) {
            Cast<float, ValT>(castBatch, batch, RoundMode::CAST_NONE, cnt * nAlignChunk);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (kUseNzTile) {
            LocalTensor<ValT> valTile = nzValBuf.Get<ValT>();
            LocalTensor<float> valFloat = nzValFloatBuf.Get<float>();
            Cast<float, ValT>(valFloat, valTile, RoundMode::CAST_NONE, cnt);
            auto evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evVS);
            WaitFlag<HardEvent::V_S>(evVS);
        }
    }

    // 把一批槽位累加进各自所属的行，用 (g0, idx0) 重放填槽时的游标推进来定位。
    __aicore__ inline void AxpyGroupBatch(LocalTensor<float> &acc, LocalTensor<ValT> &batch,
        LocalTensor<float> &castBatch, const uint32_t *rsArr, const uint32_t *reArr,
        uint32_t gCnt, uint32_t g0, uint32_t idx0, uint32_t cnt, uint32_t chunkAlign)
    {
        LocalTensor<float> valFloat;
        if constexpr (kUseNzTile) {
            valFloat = nzValFloatBuf.Get<float>();
        }
        uint32_t gg = g0;
        uint32_t ii = idx0;
        for (uint32_t j = 0; j < cnt; j++) {
            while (gg < gCnt && ii >= reArr[gg]) {
                gg++;
                if (gg < gCnt) {
                    ii = rsArr[gg];
                }
            }
            float val;
            if constexpr (kUseNzTile) {
                val = valFloat.GetValue(j) * alphaVal;
            } else {
                val = static_cast<float>(valuesGm.GetValue(ii)) * alphaVal;
            }
            if constexpr (kIsFloat) {
                Axpy(acc[gg * nAlignChunk], batch[j * nAlignChunk], val,
                    static_cast<int32_t>(chunkAlign));
            } else {
                Axpy(acc[gg * nAlignChunk], castBatch[j * nAlignChunk], val,
                    static_cast<int32_t>(chunkAlign));
            }
            ii++;
        }
    }

    __aicore__ inline void AccumulateGroup(LocalTensor<float> &acc, LocalTensor<ValT> &batch,
        LocalTensor<float> &castBatch, uint32_t *rsArr, uint32_t *reArr, uint32_t gCnt,
        uint32_t colStart, uint32_t chunkAlign, DataCopyParams copyParams,
        DataCopyPadParams padParams)
    {
        uint32_t g = 0;
        uint32_t idx = rsArr[0];
        while (g < gCnt) {
            const uint32_t g0 = g;
            const uint32_t idx0 = idx;
            uint32_t firstIdx = 0;
            const uint32_t cnt = FillGroupBatch(batch, rsArr, reArr, gCnt, g, idx, firstIdx,
                colStart, copyParams, padParams);
            if (cnt == 0) {
                break;
            }
            // bf16 的非零值整批搬进 UB 再向量 Cast（标量转换后端不支持）。调用方保证
            // 本组行号连续，于是 CSR 里这一批的 cnt 个值就是从 firstIdx 起的连续
            // cnt 个元素——游标只在行内递增、跨行时接到下一行的起点，而连续行块的
            // rowOffsets 首尾相接，中间即使有空行也不会留下空洞。
            if constexpr (kUseNzTile) {
                LocalTensor<ValT> valTile = nzValBuf.Get<ValT>();
                DataCopyParams valCopy{1, static_cast<uint16_t>(cnt * sizeof(ValT)), 0, 0};
                DataCopyPad(valTile, valuesGm[firstIdx], valCopy, padParams);
            }
            SyncAndCastBatch(batch, castBatch, cnt);
            AxpyGroupBatch(acc, batch, castBatch, rsArr, reArr, gCnt, g0, idx0, cnt, chunkAlign);

            // WAR：复写槽位前要等本批的 Axpy 读完。这里无条件发，包括本组最后一批：
            // 写回前的 V_MTE3 只挡住 MTE3，挡不住下一组填槽的 MTE2（裸 SetFlag/
            // WaitFlag 不像 TQue 的 DeQue 那样会让标量流水线停到 PIPE_V 排空）。
            // 逐行路径里同一个事件必须按行有条件发（每行一发实测 P-01 53.4->66.5us），
            // 分组后它摊到每 rowGroup 行一次，代价可以忽略。
            auto evWar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evWar);
            WaitFlag<HardEvent::V_MTE2>(evWar);
        }
    }

    // ======================== 流水化分组路径 ========================
    //
    // 语义与上面的 AccumulateGroup 完全相同（acc[g] += alpha * Σ val * B[col]），
    // 差别只在三处，都是为了让行度数小的场景摊开固定开销：
    //
    // 1) 列索引与非零值按段整块搬进 UB，热点里不再回 GM 读标量。AccumulateGroup
    //    每个非零元有两次阻塞式 GM 标量读（列索引一次、值一次，bf16 的值那次已经
    //    批量化）。这一步单独做收益有限——曾把两次读整个去掉，P-02 fp32 只从
    //    5599us 降到 4408us（21%）——它真正的意义是解开第 2 步的前置条件：col 还在
    //    GM 里时，要预取第 j+k 个 B 行必须先做一次阻塞 GM 读拿 colIndices[j+k]，
    //    加深流水无从下手（当初把 bQueue 深度从 2 加到 8 反而慢 10%，根因就在这里）。
    //    两半各自都不成立，必须一起做。
    //
    // 2) B 行槽位分 SPMM_ARCH22_PIPE_STAGES 个半区轮换，下一组的 DMA 排在本组
    //    Axpy 之前，MTE2 因此藏进 V 的执行里。AccumulateGroup 是「一批 DMA -> 等齐
    //    -> 一批 Axpy -> 等 Axpy 读完防覆写」，MTE2 与 V 完全串行。
    //
    // 3) 取 col 到发 DMA 的那条依赖链按 8 路展开，让 8 次 UB 标量读背靠背发射、
    //    延迟互相重叠。流水打通之后它就是剩下的主要标量开销。
    //
    // 走不走这条路径由 host 写在 tiling 的 pipeModeHost 里，见 spmm.h。

    // 一组行共享的 UB 句柄与形状参数。流水路径的每一层都要用这批值，逐个传参会
    // 让每个函数挂十来个形参。
    struct PipeCtx {
        LocalTensor<ValT>    stage;  // PIPE_STAGES 个半区，在 UB 里首尾相接
        LocalTensor<float>   cast;   // 半区的 fp32 暂存，单份，ValT != float only
        LocalTensor<int32_t> colUb;  // 一段非零元的列索引
        LocalTensor<ValT>    valUb;  // 一段非零元的值
        LocalTensor<float>   valF;   // 上者 Cast 成 fp32，ValT != float only
        DataCopyParams       copyParams;
        DataCopyPadParams    padParams;
        uint32_t colStart;
        uint32_t chunkAlign;
    };

    // 一次性的跨引擎排序：取一个空闲事件 ID，置位后立刻等掉。轮换用的那几个 ID
    // 已被 AllocEventID 占住（置了占用位），FetchEventID 只会返回它们之外的空闲位，
    // 所以这里不会和轮换撞 ID。
    template <HardEvent E>
    __aicore__ inline void PipeSync()
    {
        auto ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
        SetFlag<E>(ev);
        WaitFlag<E>(ev);
    }

    // 轮换用的事件 ID 必须 AllocEventID 而不是 FetchEventID：后者只返回「当前第一个
    // 空闲位」且不置占用位（见 CANN 的 kernel_tpipe_impl.h，AllocEventID 会 sbitset1、
    // FetchEventID 不会），连续两次调用拿到同一个 ID，轮换退化成单 ID，而同一个 ID
    // 上连着置位两次中间没人等，旗只是一个 bit，会吞掉一个令牌。整个 kernel 生命期
    // 都要用，故不 Release。
    __aicore__ inline void AllocPipeEvents(TPipe &pipeRef)
    {
        for (uint32_t i = 0; i < SPMM_ARCH22_PIPE_STAGES; i++) {
            evtM2V[i] = static_cast<event_t>(pipeRef.AllocEventID<HardEvent::MTE2_V>());
        }
        for (uint32_t i = 0; i + 1 < SPMM_ARCH22_PIPE_STAGES; i++) {
            evtV2M[i] = static_cast<event_t>(pipeRef.AllocEventID<HardEvent::V_MTE2>());
        }
    }

    // 把 [segBase, segBase+segLen) 这一段非零元的列索引与值整块搬进 UB，并建立后续
    // 标量读所需的跨引擎顺序。段内下标必须连续，由调用方保证（组内行号连续，或已
    // 退化成单行）。
    __aicore__ inline void LoadNzSegment(PipeCtx &pc, uint32_t segBase, uint32_t segLen)
    {
        // WAR：上一段的 colUb/valUb 标量读要全部读完，才能让 MTE2 覆写。
        PipeSync<HardEvent::S_MTE2>();

        DataCopyParams colCopy{1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0};
        DataCopyParams valCopy{1, static_cast<uint16_t>(segLen * sizeof(ValT)), 0, 0};
        DataCopyPad(pc.colUb, colIndicesGm[segBase], colCopy, pc.padParams);
        DataCopyPad(pc.valUb, valuesGm[segBase], valCopy, pc.padParams);

        // kVecRowOffset：把列号整段向量化成 B 行元素偏移 colUb <- (col - base) * N。
        // 原来这两步（减基址、乘 N、uint64 转换）在热循环里逐非零元用标量做，是标量
        // 流水的主要负担（profiler 实测 P-02 fp32 aiv_scalar_ratio=0.93）。整段一条
        // Adds/Muls 挪到向量侧后，IssueBRowAt 只剩一次 GetValue。偏移用 int32，host 侧
        // 仅在 K*N<=INT32_MAX 且 nnz 够大时置位（小矩阵回落下面的标量路径不退步）。
        // 仅窄类型启用：fp16/bf16 本就有值 Cast 的向量相位，把偏移 Muls 塞进同一相位
        // 几乎不加同步；fp32 没有向量相位，加进去反而多两次跨引擎同步，实测 −6%，故
        // fp32 恒走下面基线（IssueBRowAt 侧用标量换算），零回退。
        // 窄类型把 alpha 顺路乘进 Cast 后的 valF，避免热点逐非零元标量乘法；alpha=1
        // 时跳过这条向量指令。fp32 没有可复用的向量相位，仍在 AxpySegSlots 标量处理。
        if constexpr (!kIsFloat) {
            if (vecRowOffset) {
                PipeSync<HardEvent::MTE2_V>();
                if (idxBaseOffset != 0) {
                    Adds(pc.colUb, pc.colUb, -static_cast<int32_t>(idxBaseOffset),
                        static_cast<int32_t>(ROUNDUP(segLen, 32)));
                }
                Muls(pc.colUb, pc.colUb, static_cast<int32_t>(N),
                    static_cast<int32_t>(ROUNDUP(segLen, 32)));
                Cast<float, ValT>(pc.valF, pc.valUb, RoundMode::CAST_NONE,
                    static_cast<int32_t>(ROUNDUP(segLen, 32)));
                if (alphaVal != 1.0f) {
                    Muls(pc.valF, pc.valF, alphaVal,
                        static_cast<int32_t>(ROUNDUP(segLen, 32)));
                }
                PipeSync<HardEvent::V_S>();   // offset 与 valF 都由向量写，valUb 不再标量读
                return;
            }
        }

        // fp32 没有可复用的 Cast 相位，新开向量相位的同步成本抵消收益，继续走
        // 标量列号换算；fp16/bf16 则复用已有相位。
        if constexpr (kIsFloat) {
            PipeSync<HardEvent::MTE2_S>();
            return;
        }
        // 窄类型的值整段 Cast 成 fp32 再按标量读：bf16 的 static_cast<float> 后端
        // 不支持，half 也一并走向量，省一条分支。Cast 长度按 32 元素向上取整，尾部
        // 多算的几个是 UB 里的垃圾值，消费侧只读 [0, segLen)；SPMM_ARCH22_NZ_TILE
        // 是 32 的整数倍，所以向上取整不会越出缓冲。
        PipeSync<HardEvent::MTE2_V>();
        Cast<float, ValT>(pc.valF, pc.valUb, RoundMode::CAST_NONE,
            static_cast<int32_t>(ROUNDUP(segLen, 32)));
        if (alphaVal != 1.0f) {
            Muls(pc.valF, pc.valF, alphaVal,
                static_cast<int32_t>(ROUNDUP(segLen, 32)));
        }
        PipeSync<HardEvent::V_S>();
        // colUb 是 MTE2 写的，上面那个 V_S 只挡了 V->S，再挡一次 MTE2->S 才覆盖它。
        PipeSync<HardEvent::MTE2_S>();
    }

    // 发一发 B 行 DMA 到指定槽位偏移。列号由调用方先读好——把「读 col」与「发 DMA」
    // 拆开是 8 路展开的全部意义所在。
    __aicore__ inline void IssueBRowAt(LocalTensor<ValT> &st, uint32_t dstOff, int32_t colOrOff,
        PipeCtx &pc)
    {
        // vecRowOffset 打开时 colOrOff 已是 LoadNzSegment 整段向量化算好的 B 行元素偏移
        // (col-base)*N，热循环省掉标量减/乘/uint64 转换；关闭时（小矩阵或 K*N 可能溢出
        // int32）回落原标量算法，行为与基线一致。fp32 不做向量化偏移，即使 host 侧
        // vecRowOffset=true 也没有改写 colUb，必须按 kIsFloat 单独走原始列号换算。
        uint64_t rowBase;
        if constexpr (kIsFloat) {
            rowBase = static_cast<uint64_t>(static_cast<uint32_t>(colOrOff - idxBaseOffset)) * N;
        } else {
            rowBase = vecRowOffset
                ? static_cast<uint64_t>(static_cast<uint32_t>(colOrOff))
                : static_cast<uint64_t>(static_cast<uint32_t>(colOrOff - idxBaseOffset)) * N;
        }
        DataCopyPad(st[dstOff], bGm[rowBase + pc.colStart], pc.copyParams, pc.padParams);
    }

    // 8 路展开的一块：先把 8 个列号读出来，再连发 8 发 DMA。读一个立刻发一个的写法
    // 会让每发 DMA 都串在一次 UB 标量读的延迟后面；8 次读背靠背发射时延迟互相重叠。
    __aicore__ inline void IssueBRowsOct(LocalTensor<ValT> &st, uint32_t k, uint32_t dstOff,
        PipeCtx &pc)
    {
        const uint32_t pitchL = nAlignChunk;
        int32_t c0 = pc.colUb.GetValue(k);
        int32_t c1 = pc.colUb.GetValue(k + 1);
        int32_t c2 = pc.colUb.GetValue(k + 2);
        int32_t c3 = pc.colUb.GetValue(k + 3);
        int32_t c4 = pc.colUb.GetValue(k + 4);
        int32_t c5 = pc.colUb.GetValue(k + 5);
        int32_t c6 = pc.colUb.GetValue(k + 6);
        int32_t c7 = pc.colUb.GetValue(k + 7);
        IssueBRowAt(st, dstOff, c0, pc);
        IssueBRowAt(st, dstOff + pitchL, c1, pc);
        IssueBRowAt(st, dstOff + 2 * pitchL, c2, pc);
        IssueBRowAt(st, dstOff + 3 * pitchL, c3, pc);
        IssueBRowAt(st, dstOff + 4 * pitchL, c4, pc);
        IssueBRowAt(st, dstOff + 5 * pitchL, c5, pc);
        IssueBRowAt(st, dstOff + 6 * pitchL, c6, pc);
        IssueBRowAt(st, dstOff + 7 * pitchL, c7, pc);
    }

    // 连发 cnt 发 B 行 DMA 到 st 这个半区的前 cnt 个槽位；ubIdx 是本组第一个非零元
    // 在 colUb 里的下标。
    __aicore__ inline void IssueBRows(LocalTensor<ValT> &st, uint32_t ubIdx, uint32_t cnt,
        PipeCtx &pc)
    {
        const uint32_t pitchL = nAlignChunk;
        uint32_t dstOff = 0;
        uint32_t j = 0;
        for (; j + 8 <= cnt; j += 8) {
            IssueBRowsOct(st, ubIdx + j, dstOff, pc);
            dstOff += 8 * pitchL;
        }
        for (; j < cnt; j++) {
            IssueBRowAt(st, dstOff, pc.colUb.GetValue(ubIdx + j), pc);
            dstOff += pitchL;
        }
    }

    // 把一个半区里的槽位累加进所属输出行。具体循环拆到小 helper，避免这里重新
    // 引入头文件长度、函数长度、圈复杂度和嵌套深度门禁问题。
    __aicore__ inline void AxpySegSlots(LocalTensor<float> &acc, PipeCtx &pc,
        const uint32_t *reArr, uint32_t gCnt, uint32_t segBase, uint32_t ubIdx, uint32_t cnt,
        uint32_t sel, uint32_t &gCur)
    {
        SpmmAxpySegmentSlots<ValT>(acc, pc, reArr, gCnt, segBase, ubIdx, cnt,
            sel, nAlignChunk, stageStride, alphaVal, gCur);
    }

    // 一段非零元的软件流水规约。每一轮的顺序是
    //   等本组 DMA 到位 -> 等目标半区可写 -> 发下一组 DMA -> 做本组 Axpy，
    // 于是下一组的搬运与本组的向量执行重叠。
    //
    // 令牌配平（n = 组数，S = PIPE_STAGES，nVEvt = S-1）：
    //   MTE2->V：第 0 组的令牌在序头置于 ID 0，第 g 组（g>=1）置于第 g-1 轮体内的
    //     ID (g mod S)；第 p 轮等 ID (p mod S)。共 n 置 n 等。必须给满 S 个 ID 而
    //     不是 2 个——第 g+S 组的置位位于第 g+S-1 轮体内，其前面挡着 V->MTE2 的等待，
    //     而那面旗由第 g 轮 Axpy 之后置出，所以 V 必然已经吃掉了第 g 组的令牌；
    //     只用 2 个 ID 时 MTE2 会冲过序头那两次「免检」等待，把令牌吞掉并最终死锁。
    //   V->MTE2：序头给每个 ID 预置一面旗（共 nVEvt 面），第 p 轮先等 ID
    //     (p mod nVEvt)、Axpy 之后置回同一个 ID；退出时每个 ID 各余一面，收尾等掉。
    //     第 i 轮要覆写的是半区 (i+1) mod S，而它上一次被读是在第 i-nVEvt 轮，
    //     (i-nVEvt) mod S 正等于 (i+1) mod S —— 等的是上上一组，早就做完了。
    __aicore__ inline void RunPipedSegment(LocalTensor<float> &acc, PipeCtx &pc,
        const uint32_t *reArr, uint32_t gCnt, uint32_t segBase, uint32_t segLen, uint32_t &gCur)
    {
        constexpr uint32_t nStage = SPMM_ARCH22_PIPE_STAGES;
        constexpr uint32_t nVEvt = nStage - 1;
        const uint32_t slots = batchSlots;

        // 序头。先把流水排空：上一段的 Axpy 或 InitGroupAcc 的 Muls 都可能还在读
        // 半区，而下面就要往里发 DMA 了。
        PipeSync<HardEvent::V_MTE2>();
        for (uint32_t i = 0; i < nVEvt; i++) {
            SetFlag<HardEvent::V_MTE2>(evtV2M[i]);
        }
        const uint32_t cnt0 = (segLen < slots) ? segLen : slots;
        LocalTensor<ValT> st0 = pc.stage;
        IssueBRows(st0, 0, cnt0, pc);
        SetFlag<HardEvent::MTE2_V>(evtM2V[0]);

        uint32_t sel = 0;
        uint32_t selNext = 1;
        uint32_t vi = 0;
        for (uint32_t p = 0; p < segLen; p += slots) {
            uint32_t cnt = segLen - p;
            if (cnt > slots) {
                cnt = slots;
            }
            WaitFlag<HardEvent::MTE2_V>(evtM2V[sel]);
            WaitFlag<HardEvent::V_MTE2>(evtV2M[vi]);
            IssueNextPipeGroup(pc, p, segLen, slots, selNext);
            if constexpr (!kIsFloat) {
                // cnt 个槽位在 UB 里首尾相接，整组一条 Cast。
                Cast<float, ValT>(pc.cast, pc.stage[sel * stageStride], RoundMode::CAST_NONE,
                    static_cast<int32_t>(cnt * nAlignChunk));
                PipeBarrier<PIPE_V>();
            }
            AxpySegSlots(acc, pc, reArr, gCnt, segBase, p, cnt, sel, gCur);
            SetFlag<HardEvent::V_MTE2>(evtV2M[vi]);
            sel = selNext;
            selNext = (selNext + 1 >= nStage) ? 0 : (selNext + 1);
            vi = (vi + 1 >= nVEvt) ? 0 : (vi + 1);
        }
        // 收尾：每个 V->MTE2 的 ID 上都还余一面旗，等掉，令牌归零。
        for (uint32_t i = 0; i < nVEvt; i++) {
            WaitFlag<HardEvent::V_MTE2>(evtV2M[i]);
        }
    }

    // 发下一组的 B 行 DMA。这是软件流水的关键：它排在本组 Axpy 之前，MTE2 因此
    // 藏进 V 的执行里。最后一组之后没有下一组，什么都不做。
    __aicore__ inline void IssueNextPipeGroup(PipeCtx &pc, uint32_t p, uint32_t segLen,
        uint32_t slots, uint32_t selNext)
    {
        const uint32_t next = p + slots;
        if (next >= segLen) {
            return;
        }
        uint32_t cntN = segLen - next;
        if (cntN > slots) {
            cntN = slots;
        }
        LocalTensor<ValT> stN = pc.stage[selNext * stageStride];
        IssueBRows(stN, next, cntN, pc);
        SetFlag<HardEvent::MTE2_V>(evtM2V[selNext]);
    }

    // 一组行（或退化时的一行）的整体规约，按 SPMM_ARCH22_NZ_TILE 切段推进。
    // 组内行号连续时整组的非零元在 CSR 里就是 [rsArr[0], reArr[gCnt-1]) 一整段，
    // 中间即使有空行也不留空洞，所以段内下标必然连续、可以整块 DMA。
    __aicore__ inline void AccumulateGroupPiped(LocalTensor<float> &acc, PipeCtx &pc,
        const uint32_t *rsArr, const uint32_t *reArr, uint32_t gCnt)
    {
        const uint32_t nzBegin = rsArr[0];
        const uint32_t nzEnd = reArr[gCnt - 1];
        if (nzBegin >= nzEnd) {
            return;
        }
        uint32_t gCur = 0;
        for (uint32_t segBase = nzBegin; segBase < nzEnd; segBase += SPMM_ARCH22_NZ_TILE) {
            uint32_t segLen = nzEnd - segBase;
            if (segLen > SPMM_ARCH22_NZ_TILE) {
                segLen = SPMM_ARCH22_NZ_TILE;
            }
            LoadNzSegment(pc, segBase, segLen);
            RunPipedSegment(acc, pc, reArr, gCnt, segBase, segLen, gCur);
        }
    }

    // 流水路径的组内分发：行号不连续时按行退化——段的整块 DMA 要求段内下标连续，
    // 跨行只在行号连续时成立，行内则必然连续。C 的读入/写出与 Duplicate/Muls 仍按
    // 整组走，只有累加退化，与 AccumulateGroup 那条 kUseNzTile 分支同理。
    __aicore__ inline void AccumulateGroupPipedDispatch(LocalTensor<float> &acc, PipeCtx &pc,
        const GroupCtx &gc)
    {
        if (gc.contig) {
            AccumulateGroupPiped(acc, pc, grpRs, grpRe, gc.gCnt);
            return;
        }
        for (uint32_t g = 0; g < gc.gCnt; g++) {
            LocalTensor<float> accRow = acc[g * nAlignChunk];
            AccumulateGroupPiped(accRow, pc, grpRs + g, grpRe + g, 1);
        }
    }

    // 建好流水路径要用的 UB 句柄与形状参数。每个列分块调一次，不进热点。
    __aicore__ inline void BuildPipeCtx(PipeCtx &pc, LocalTensor<ValT> &batch,
        LocalTensor<float> &castBatch, uint32_t colStart, uint32_t chunkAlign,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        pc.stage = batch;
        pc.colUb = nzIdxBuf.Get<int32_t>();
        pc.valUb = nzValBuf.Get<ValT>();
        if constexpr (!kIsFloat) {
            pc.cast = castBatch;
            pc.valF = nzValFloatBuf.Get<float>();
        }
        pc.copyParams = copyParams;
        pc.padParams = padParams;
        pc.colStart = colStart;
        pc.chunkAlign = chunkAlign;
    }

    GlobalTensor<int32_t> colIndicesGm;
    GlobalTensor<ValT>    valuesGm;
    GlobalTensor<ValT>    bGm;
    GlobalTensor<ValT>    cGm;
    GlobalTensor<ValT>    yGm;

    TBuf<TPosition::VECCALC> nzIdxBuf;       // 列索引整批 tile，逐行整批路径与流水路径共用
    TBuf<TPosition::VECCALC> nzValBuf;       // 非零值整批 tile
    TBuf<TPosition::VECCALC> nzValFloatBuf;  // 上者 Cast 成 fp32 的落地，ValT != float only
    TBuf<TPosition::VECCALC> bBatchBuf;      // B 行槽位，分组路径与逐行整批路径共用
    TBuf<TPosition::VECCALC> bCastBuf;       // 槽位的 fp32 暂存，ValT != float only
    TBuf<TPosition::VECCALC> accGroupBuf;    // rowGroup 份 fp32 累加器
    TBuf<TPosition::VECCALC> stageGroupBuf;  // 同宽的 ValT 暂存，ValT != float only

    float alphaVal, betaVal;

    // 流水路径的状态。stageStride 是一个半区的元素数；两组事件 ID 由
    // AllocPipeEvents 用 AllocEventID 占住后按半区轮换，见 RunPipedSegment。
    uint32_t stageStride;
    event_t evtM2V[SPMM_ARCH22_PIPE_STAGES];
    event_t evtV2M[SPMM_ARCH22_PIPE_STAGES - 1];
};
