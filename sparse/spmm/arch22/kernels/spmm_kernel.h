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
#include "spmm_kernel_common.h"
#include "spmm_kernel_group.h"
#include "spmm_kernel_entry_macros.h"

// 本头文件必须与各 dtype 的 .cpp 编译单元一起放在 kernels/ 目录下。那些编译单元
// 以同目录相对路径引用本文件，而编译器解析相对路径时按字面拼接：若本文件被移到
// 上一层 arch22/，本文件里指向上一层的相对路径就会被多拼一层，落到 arch22 的上级
// 目录，从而找不到 spmm.h。

// AscendC 具体名字引入集中在此共享头（using-declaration，非 using-directive；见其内
// 说明）。本文件及其后 #include 的 spmm_kernel_generic.h 都依赖这些名字。
#include "spmm_kernel_using.h"

// TQue 的模板参数是可分配缓冲的上界；实际深度由 InitBuffer 的运行时入参给出，
// 见 InitLocalBuffers 里的 SpmmArch22PrefetchDepth。
constexpr uint32_t SPMM_KERNEL_BQUEUE_MAX = SPMM_ARCH22_BQUEUE_MAX;

/**
 * @brief CSR 稀疏矩阵 x 稠密矩阵乘法（SpMM）AscendC 算子，按值类型模板化。
 *
 * 行并行 + 列分块流式处理：每个 AI Core 负责若干整行（由 host 端 LPT 装箱决定），
 * 每行按列 chunk 遍历，chunk 内以稠密矩阵 B 的行（= 稀疏矩阵的列）为粒度做
 * DataCopyPad 双缓冲预取 + Muls/Add 规约。
 *
 * 计算精度：无论 ValT 是 float/half/bfloat16_t，规约累加(accLocal)始终用 float
 * 完成，仅在读入 B/C、写出 Y 时按需 Cast，满足任务书"fp16/bf16 用 fp32 累加"的
 * 精度要求。ValT==float 时直接复用 outQueue 的 tile 做累加，不引入额外 Cast/
 * 额外 UB 占用，与合入基线（xssxyz, ops-sparse!92）的原始行为逐位一致。
 *
 * 本文件只留逐行路径与初始化；跨行分组路径在基类 SpmmArch22GroupPath 里，两者
 * 通过 ProcessChunk 的一次分发相接。
 *
 * @tparam ValT 稀疏矩阵非零值 / 稠密矩阵 B / 稠密矩阵 C 的存储类型（三者当前要求一致）。
 */
template <typename ValT>
class SpmmArch22Kernel : public SpmmArch22GroupPath<ValT> {
    // Init()/Process() 入口样板与 complex64 版逐字一致，收敛到共享宏（见
    // spmm_kernel_entry_macros.h）；实数版非 generic 分支无额外初始化，POST 传空。
    SPMM_ARCH22_DEFINE_KERNEL_ENTRY()

private:
    // 基类的列分块遍历需要回调下面 private 的 ProcessChunk。
    friend class SpmmArch22KernelCommon;

    // 基类依赖 ValT，于是本类里对继承成员的非限定引用都会在两阶段名字查找的第一阶段
    // 失败（编译期还不知道 Base 的成员集合）。惯用解法有两个：逐处写 this->x，或者
    // 用 using 把这些名字重新引入本类作用域。这里选后者——函数体保持原样，成员清单
    // 集中在一处可查，也不会因为漏加 this-> 而留下只在某个 dtype 实例化时才炸的坑。
    using Base = SpmmArch22GroupPath<ValT>;
    using Base::kIsFloat;
    using Base::kUseNzTile;
    using Base::InitCommonTiling;
    using Base::InitRowRange;
    using Base::ProcessChunks;
    using Base::ProcessChunkGrouped;
    using Base::M;
    using Base::N;
    using Base::nnz;
    using Base::nMaxPerChunk;
    using Base::nAlignChunk;
    using Base::myRowStart;
    using Base::myRowEnd;
    using Base::skipBeta;
    using Base::idxBaseOffset;
    using Base::batchSlots;
    using Base::rowGroup;
    using Base::fuseGroupDma;
    using Base::pipeMode;
    using Base::stageStride;
    using Base::AllocPipeEvents;
    using Base::reorderGm;
    using Base::rowOffsetsGm;
    using Base::rowOffBuf;
    using Base::colIndicesGm;
    using Base::valuesGm;
    using Base::bGm;
    using Base::cGm;
    using Base::yGm;
    using Base::nzIdxBuf;
    using Base::nzValBuf;
    using Base::nzValFloatBuf;
    using Base::bBatchBuf;
    using Base::bCastBuf;
    using Base::accGroupBuf;
    using Base::stageGroupBuf;
    using Base::alphaVal;
    using Base::betaVal;
    using Base::SyncAndCastBatch;

    // 与值类型无关的字段由基类的 InitCommonTiling 读取（M/N/nnz/nChunks/
    // idxBaseOffset），这里只补实数版特有的 K 和 alpha/beta，并据 beta 定 skipBeta。
    __aicore__ inline void InitTiling(__gm__ SpmmArch22TilingData *td)
    {
        InitCommonTiling(td);
        K = td->K;
        alphaVal = td->alphaHost;
        betaVal = td->betaHost;
        ldb = td->ldbHost;
        ldc = td->ldcHost;
        bOrder = td->bOrderHost;
        cOrder = td->cOrderHost;
        aRows = td->aRowsHost;
        opA = td->opAHost;
        opB = td->opBHost;
        genericPath = (td->genericPathHost != 0);
        highPrecision = kIsFloat &&
            td->algHost == static_cast<int32_t>(2);
        // beta==0 时整段跳过读 C，与 torch.sparse.addmm 的短路语义一致，也不再
        // 依赖 Muls(·, 0.0f) 在向量单元上是否传播 NaN 这个未言明的行为。
        skipBeta = (betaVal == 0.0f);
    }

    // Public SpMM passes the same pointer for C/Y; the ATen fast path may keep
    // the dense input separate from the freshly allocated output.
    __aicore__ inline uint64_t StorageElems(
        uint32_t rows, uint32_t cols, int32_t ld, int32_t order)
    {
        if (rows == 0 || cols == 0) {
            return 0;
        }
        return order == 0 ?
            static_cast<uint64_t>(rows - 1) * ld + cols :
            static_cast<uint64_t>(cols - 1) * ld + rows;
    }

    __aicore__ inline void InitGlobalBuffers(__gm__ SpmmArch22TilingData *td,
        GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values, GM_ADDR matB, GM_ADDR matC,
        GM_ADDR matY, GM_ADDR workspace)
    {
        rowOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowOff), aRows + 1);
        colIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(colIdx), nnz);
        valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(values), nnz);
        const uint64_t bElems = StorageElems(td->bRowsHost, td->bColsHost, ldb, bOrder);
        const uint64_t cElems = StorageElems(M, N, ldc, cOrder);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(matB), bElems);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(matC), cElems);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ ValT *>(matY), cElems);
        SPMM_ARCH22_BIND_TRANSPOSE_GM(td, workspace);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        // UB 预算：bQueue 预取环(bDepth 份 ValT) + outQueue 1 份(ValT) + tmpBuf 1 份
        // (float) + (ValT!=float 时) accBuf 1 份(float) + bFloatBuf 1 份(float) 做
        // Cast 暂存。ValT==float 时 accLocal 直接借用 outQueue 的 tile，不需要
        // accBuf/bFloatBuf，与基线一致。
        //
        // 每元素字节数与 nMax 都由 spmm.h 里的共用函数算出，host 侧
        // SpmmArch22ComputeNChunks 调的是同一组函数，两侧口径不可能漂移。
        constexpr uint32_t valBytes = sizeof(ValT);
        bDepth = SpmmArch22PrefetchDepth(N, valBytes);
        uint64_t nMax = SpmmArch22NMaxForDepth(valBytes, bDepth);
        if (nMax > N) {
            nMax = N;
        }
        nMaxPerChunk = static_cast<uint32_t>(nMax);
        // 对齐粒度按 valBytes 反推，与 host 的 SpmmArch22NAlignChunk 同源。窄类型
        // 下写死 SPMM_ARCH22_ALIGN 会让槽位步长只有 16 字节，见 SpmmArch22AlignElems。
        nAlignChunk = ROUNDUP(nMaxPerChunk, SpmmArch22AlignElems(valBytes));

        pipe.InitBuffer(bQueue, bDepth, nAlignChunk * sizeof(ValT));
        pipe.InitBuffer(outQueue, 1, nAlignChunk * sizeof(ValT));
        pipe.InitBuffer(tmpBuf, nAlignChunk * sizeof(float));
        if constexpr (!kIsFloat) {
            pipe.InitBuffer(accBuf, nAlignChunk * sizeof(float));
            pipe.InitBuffer(bFloatBuf, nAlignChunk * sizeof(float));
        }
        // 这三块定长(与 N 无关)tile 有两个消费者：逐行整批路径 AccumulateRowTiled
        // (只有 bf16 走)，以及流水分组路径的段缓存(所有 dtype 都走)。两者不会同时
        // 运行——一个在 rowGroup==0 时才被调到，另一个要求 rowGroup>0——所以复用同
        // 一组缓冲。它们计入 SPMM_ARCH22_UB_OVERHEAD 而不是 bytesPerElem，所以 host
        // 侧 SpmmArch22ComputeNChunks 的每元素口径不必跟着改：最宽的情形是窄类型的
        // 4+2+4=10 字节/非零元，256 个非零元占 2.5KB，仍在预留的 4KB 之内。
        if (kUseNzTile || pipeMode) {
            pipe.InitBuffer(nzIdxBuf, SPMM_ARCH22_NZ_TILE * sizeof(int32_t));
            pipe.InitBuffer(nzValBuf, SPMM_ARCH22_NZ_TILE * sizeof(ValT));
            if constexpr (!kIsFloat) {
                pipe.InitBuffer(nzValFloatBuf, SPMM_ARCH22_NZ_TILE * sizeof(float));
            }
        }

        InitGroupBuffers(valBytes);
    }

    // 整批槽位与行分组的缓冲。与上面那批的区别是它们只花"按 N 伸缩的预算之外剩下的
    // UB"，能放多少放多少，因此不影响 nChunks；拆成单独一个函数是为了让两边各自
    // 保持在门禁的函数长度上限以内。
    __aicore__ inline void InitGroupBuffers(uint32_t valBytes)
    {
        // AccumulateRowBatched 的槽位数组。它按 N 伸缩，所以不能计入
        // SPMM_ARCH22_UB_OVERHEAD；这里不改 host 侧的每元素口径（那会改变 nChunks），
        // 而是用 UB 里剩下的余量能放多少放多少，放不下 SLOTS_MIN 个就退回逐元素
        // 路径。N 大的时候本来也不太需要这条路径：单次 B 行搬运越接近连续大块传输，
        // 逐元素簿记的占比越低。
        // 每个槽位要 valBytes；窄类型还要一份等宽的 fp32 暂存供整批 Cast 落地。
        //
        // 流水模式下槽位要 SPMM_ARCH22_PIPE_STAGES 份（半区在同一块 UB 里首尾相接），
        // 因此换用 SpmmArch22PipeSlots 的口径；host 侧决定装箱粒度时走的是配套的
        // SpmmArch22PipeRowGroup，两侧同源。窄类型的 fp32 暂存仍只要一份——它在组内
        // 被 Cast 写、被 Axpy 读，PIPE_V 内部有序，不需要轮换。
        batchSlots = pipeMode ? SpmmArch22PipeSlots(N, valBytes)
                              : SpmmArch22BatchSlots(N, valBytes);
        stageStride = 0;
        // 流水路径拿 batchSlots 当步长推进，0 会让循环原地打转。host 只在
        // PipeRowGroup>0（蕴含 PipeSlots>0）时才打开开关，这里再兜一次。
        if (batchSlots == 0) {
            pipeMode = false;
        }
        if (batchSlots > 0) {
            stageStride = batchSlots * nAlignChunk;
            const uint32_t stages = pipeMode ? SPMM_ARCH22_PIPE_STAGES : 1;
            pipe.InitBuffer(bBatchBuf,
                static_cast<uint32_t>(stageStride * stages * valBytes));
            if constexpr (!kIsFloat) {
                pipe.InitBuffer(bCastBuf,
                    static_cast<uint32_t>(stageStride * sizeof(float)));
            }
        }

        // 行分组的累加器。同样只花 UB 余量，不进 host 侧的每元素口径。
        // 每组行要一份 fp32 累加器；窄类型还要一份等宽的 ValT 暂存，用于把 C 读进来
        // 和把结果 Cast 出去（fp32 直接拿累加器当 DMA 的源和目的，不需要暂存）。
        //
        // host 侧 DoPreProcess 调同一个函数决定连续行块的粒度，见 SpmmArch22RowGroup。
        //
        // 三种实数 dtype 都走这条路径，bf16 也在内。它的非零值读不了标量（见
        // kUseNzTile），只能整批 DMA 进 UB 再向量 Cast，这一度让它无法分组——跨行的
        // 一批在 values 数组里不连续。连续行块装箱之后障碍就没了：一组的行号连续，
        // CSR 里它们的非零值正好是 values[off[r0], off[r0+gCnt]) 一整段，而游标是
        // 单调推进的，所以每批的 cnt 个值必然是连续的 cnt 个元素，一次 DMA 取回。
        rowGroup = pipeMode ? SpmmArch22PipeRowGroup(N, valBytes)
                            : SpmmArch22RowGroup(N, valBytes);
        if (rowGroup > 0) {
            pipe.InitBuffer(accGroupBuf,
                static_cast<uint32_t>(rowGroup * nAlignChunk * sizeof(float)));
            if constexpr (!kIsFloat) {
                pipe.InitBuffer(stageGroupBuf,
                    static_cast<uint32_t>(rowGroup * nAlignChunk * valBytes));
            }
            // 连续行块时整组的 rowOffsets 一次搬进来，见 ProcessChunkGrouped。定长
            // (与 N 无关)，计入 SPMM_ARCH22_UB_OVERHEAD 的余量。
            pipe.InitBuffer(rowOffBuf, (SPMM_ARCH22_ROW_GROUP_MAX + 1) * sizeof(int32_t));
            // 整组的 C 读入/写出能否合成一次连续 DMA：需要单 chunk 覆盖整个 N，
            // 且 UB 里槽位间没有对齐补白，否则 GM 侧与 UB 侧的步长都不为零，
            // DataCopyPad 的 stride 语义(UB 侧按 32 字节块计)容易出错，不值得冒险。
            fuseGroupDma = (nAlignChunk == N);
        } else {
            fuseGroupDma = false;
        }

        // 轮换用的事件 ID 要占位，必须在这里一次性拿好，见 AllocPipeEvents 的说明。
        // 排在所有 InitBuffer 之后：TQue 自己也从同一批事件池里取 ID，先让它们占完，
        // 剩下的才轮到这里，两边不会撞。
        if (pipeMode) {
            AllocPipeEvents(pipe);
        }
    }

    // 逐行路径的选路：bf16 走整批搬运，float/half 走逐元素标量。判据 kUseNzTile
    // 在基类，那里有实测数据。
    __aicore__ inline void AccumulateRow(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t chunkAlign,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        if (batchSlots > 0) {
            AccumulateRowBatched(accLocal, tmp, rs, re, colStart, chunkAlign, copyParams, padParams);
        } else if constexpr (kUseNzTile) {
            AccumulateRowTiled(accLocal, tmp, rs, re, colStart, chunkAlign, copyParams, padParams);
        } else {
            AccumulateRowScalar(accLocal, tmp, rs, re, colStart, chunkAlign, copyParams, padParams);
        }
    }

    // 一批 batchSlots 个非零元共用一组同步事件。这是本 kernel 小 N 性能的主要来源。
    //
    // DMA 次数与 AccumulateRowScalar 完全相同，唯一的差别是不再走 TQue：那条路径
    // 每个非零元都要 AllocTensor/EnQue/DeQue/FreeTensor，每次都带跨引擎事件。定位
    // 过程排除了另外三个嫌疑：把列索引/值的标量 GM 读整个去掉只快 21%(P-02 fp32
    // 5599->4408us)、DataCopy 换掉 DataCopyPad 无差别(5753us)、把预取队列从 2 加深
    // 到 8 反而慢 10%(6032us)。剩下能解释每非零元 185ns 的就只有这笔逐元素簿记，
    // 改成整批后 P-02 fp32 降到 3641us(1.54x)、fp16 3231us(1.72x)。
    //
    // 一批内：连发 cnt 次 B 行 DMA 进同一块 UB 的 cnt 个槽位，一个 MTE2_V 事件等齐，
    // 再连做 cnt 次累加，批间一个 V_MTE2 事件防覆写。同步事件数从每非零元 2 个降到
    // 每批最多 2 个；行度数小于槽位数时(P-01/P-02/CTL 都是)每行只有一批，就只剩 1 个。
    // 同 AxpyGroupBatch，但一批槽位全属同一行：累加目标固定，值的 CSR 下标连续。
    __aicore__ inline void AxpyRowBatch(LocalTensor<float> &accLocal, LocalTensor<ValT> &batch,
        LocalTensor<float> &castBatch, uint32_t base, uint32_t cnt, uint32_t chunkAlign)
    {
        LocalTensor<float> valFloat;
        if constexpr (kUseNzTile) {
            valFloat = nzValFloatBuf.template Get<float>();
        }
        for (uint32_t j = 0; j < cnt; j++) {
            float val;
            if constexpr (kUseNzTile) {
                val = valFloat.GetValue(j) * alphaVal;
            } else {
                val = static_cast<float>(valuesGm.GetValue(base + j)) * alphaVal;
            }
            // 两条分支都是 fp32 Axpy，只是源分别取原始槽位和 Cast 后的槽位。
            if constexpr (kIsFloat) {
                Axpy(accLocal, batch[j * nAlignChunk], val, static_cast<int32_t>(chunkAlign));
            } else {
                Axpy(accLocal, castBatch[j * nAlignChunk], val,
                    static_cast<int32_t>(chunkAlign));
            }
        }
    }

    __aicore__ inline void AccumulateRowBatched(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t chunkAlign,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        LocalTensor<ValT> batch = bBatchBuf.template Get<ValT>();
        LocalTensor<float> castBatch;
        if constexpr (!kIsFloat) {
            castBatch = bCastBuf.template Get<float>();
        }

        for (uint32_t base = rs; base < re; base += batchSlots) {
            uint32_t cnt = re - base;
            if (cnt > batchSlots) {
                cnt = batchSlots;
            }

            for (uint32_t j = 0; j < cnt; j++) {
                uint32_t col =
                    static_cast<uint32_t>(colIndicesGm.GetValue(base + j) - idxBaseOffset);
                DataCopyPad(batch[j * nAlignChunk],
                    bGm[static_cast<uint64_t>(col) * N + colStart], copyParams, padParams);
            }
            // bf16 的非零值也整批搬进 UB：它的标量 static_cast<float> 后端不支持
            // (float/half 可以，见 AccumulateRowScalar)，只能靠向量 Cast。搭在同一批
            // DMA 与同一个 MTE2_V 事件上，不额外增加同步。
            if constexpr (kUseNzTile) {
                LocalTensor<ValT> valTile = nzValBuf.template Get<ValT>();
                DataCopyParams valCopy{1, static_cast<uint16_t>(cnt * sizeof(ValT)), 0, 0};
                DataCopyPad(valTile, valuesGm[base], valCopy, padParams);
            }
            SyncAndCastBatch(batch, castBatch, cnt);
            AxpyRowBatch(accLocal, batch, castBatch, base, cnt, chunkAlign);

            // WAR：复写槽位前要等本批的 Axpy(PIPE_V) 读完。必须无条件发，跨行的那
            // 一次也要——StoreRow 里 outQueue 的 EnQue/DeQue 只是一对 V->MTE3，挡住
            // 的是 MTE3，挡不住下一行填槽的 MTE2；曾按"本行还有下一批"有条件发，结果
            // 在 rowGroup==0 而 batchSlots>0 的 N 区间(fp32 的 N∈[2112,2304]、
            // [3328,3840])跨行覆写，输出既错又每次不同。同 AxpyGroupBatch 后的处理。
            auto evWar =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evWar);
            WaitFlag<HardEvent::V_MTE2>(evWar);
        }
    }

    // 发起一次 B 行搬运。列索引从 GM 标量读；调用者负责保证 bQueue 还有空槽。
    //
    // 这里试过用 DataCopy 替 DataCopyPad（N=128/256 的一行本就是 32 字节整数倍，
    // 不需要补齐能力），P-02 fp32 5599->5753us，无收益。说明固定开销来自发起一次
    // DMA 本身，不是 Pad 变体的补齐逻辑。
    __aicore__ inline void IssueBRow(uint32_t idx, uint32_t colStart,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t col = static_cast<uint32_t>(colIndicesGm.GetValue(idx) - idxBaseOffset);
        LocalTensor<ValT> b = bQueue.AllocTensor<ValT>();
        DataCopyPad(b, bGm[static_cast<uint64_t>(col) * N + colStart], copyParams, padParams);
        bQueue.EnQue(b);
    }

    // acc += val * b，算完归还槽位。用一条 Axpy 完成，而不是 Muls 到 tmp 再 Add：
    // 语义完全等价(Axpy(dst,src,scalar,count): dst[i] += scalar*src[i])，但每个
    // 非零元的向量指令数从 2 条降到 1 条。窄类型没有混合精度的 Axpy 重载，只能先
    // Cast 成 fp32 再走 Muls+Add 两条。
    __aicore__ inline void AxpyBRow(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        LocalTensor<ValT> &bCur, float val, uint32_t chunkAlign)
    {
        if constexpr (kIsFloat) {
            Axpy(accLocal, bCur, val, static_cast<int32_t>(chunkAlign));
        } else {
            LocalTensor<float> bFloat = bFloatBuf.Get<float>();
            Cast<float, ValT>(bFloat, bCur, RoundMode::CAST_NONE, chunkAlign);
            PipeBarrier<PIPE_V>();
            Muls(tmp, bFloat, val, chunkAlign);
            Add(accLocal, accLocal, tmp, chunkAlign);
        }
        bQueue.FreeTensor(bCur);
    }

    // 原逐元素路径：每个非零元一次 colIndicesGm/valuesGm.GetValue()。
    // half 的 static_cast<float> 是后端支持的标量转换；bf16 不是（见 kUseNzTile）。
    //
    // prologue 先发 bDepth-1 个 B 行搬运，循环体内每消费一个就补发一个，稳态下
    // 始终有 bDepth-1 个 DMA 在飞。留一个空槽而不是发满 bDepth 个，是为了让补发
    // 排在向量计算之前——小 DMA 的瓶颈是发起延迟，越早发出越好。
    __aicore__ inline void AccumulateRowScalar(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t chunkAlign,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t issue = rs;
        for (uint32_t k = 1; k < bDepth && issue < re; k++, issue++) {
            IssueBRow(issue, colStart, copyParams, padParams);
        }

        for (uint32_t idx = rs; idx < re; idx++) {
            LocalTensor<ValT> bCur = bQueue.DeQue<ValT>();
            float val = static_cast<float>(valuesGm.GetValue(idx)) * alphaVal;

            if (issue < re) {
                IssueBRow(issue, colStart, copyParams, padParams);
                issue++;
            }

            AxpyBRow(accLocal, tmp, bCur, val, chunkAlign);
        }
    }

    // 同 IssueBRow，但列索引从已搬进 UB 的 tile 里读，不再回 GM。
    __aicore__ inline void IssueBRowFromTile(LocalTensor<int32_t> &idxTile, uint32_t j,
        uint32_t colStart, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t col = static_cast<uint32_t>(idxTile.GetValue(j) - idxBaseOffset);
        LocalTensor<ValT> b = bQueue.AllocTensor<ValT>();
        DataCopyPad(b, bGm[static_cast<uint64_t>(col) * N + colStart], copyParams, padParams);
        bQueue.EnQue(b);
    }

    // 把一批非零元的列索引与值整块搬进 UB，并建立后续标量读所需的跨引擎顺序。
    __aicore__ inline void LoadNzTile(LocalTensor<int32_t> &idxTile, LocalTensor<ValT> &valTile,
        LocalTensor<float> &valFloat, uint32_t tileBase, uint32_t tileLen,
        DataCopyPadParams padParams)
    {
        DataCopyParams idxCopy{1, static_cast<uint16_t>(tileLen * sizeof(int32_t)), 0, 0};
        DataCopyParams valCopy{1, static_cast<uint16_t>(tileLen * sizeof(ValT)), 0, 0};
        DataCopyPad(idxTile, colIndicesGm[tileBase], idxCopy, padParams);
        DataCopyPad(valTile, valuesGm[tileBase], valCopy, padParams);

        // MTE2 写 idxTile -> S 读 idxTile。裸 TBuf 没有 TQue 的依赖跟踪，
        // 跨引擎顺序必须手动建立（PipeBarrier 只在单条流水线内部排序）。
        auto evtIdx = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evtIdx);
        WaitFlag<HardEvent::MTE2_S>(evtIdx);

        if constexpr (!kIsFloat) {
            // 窄类型：整批 Cast 成 fp32 后再按标量读。MTE2 写 valTile -> V 读，
            // V 写 valFloat -> S 读。
            auto evtToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evtToV);
            WaitFlag<HardEvent::MTE2_V>(evtToV);
            Cast<float, ValT>(valFloat, valTile, RoundMode::CAST_NONE, tileLen);
            auto evtToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evtToS);
            WaitFlag<HardEvent::V_S>(evtToS);
        }
    }

    // 整批路径：列索引和非零值以 SPMM_ARCH22_NZ_TILE 为批整块 DMA 进 UB，内层循环
    // 从 UB 读。bf16 的收益来自把"每个非零元一次 2 字节 DMA + 一次向量 Cast + 三个
    // 跨引擎同步事件"压缩成"每 256 个非零元一次 tile DMA + 一次 Cast"。
    __aicore__ inline void AccumulateRowTiled(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t chunkAlign,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        // nzIdxBuf 现在来自依赖基类（与流水路径的段缓存共用），非限定引用要加
        // template 才能把 Get 当成模板名，见类顶部 kUsingBase 附近的说明。
        LocalTensor<int32_t> idxTile = nzIdxBuf.template Get<int32_t>();
        LocalTensor<ValT> valTile = nzValBuf.template Get<ValT>();
        LocalTensor<float> valFloat;
        if constexpr (!kIsFloat) {
            valFloat = nzValFloatBuf.template Get<float>();
        }

        for (uint32_t tileBase = rs; tileBase < re; tileBase += SPMM_ARCH22_NZ_TILE) {
            uint32_t tileLen = re - tileBase;
            if (tileLen > SPMM_ARCH22_NZ_TILE) {
                tileLen = SPMM_ARCH22_NZ_TILE;
            }

            LoadNzTile(idxTile, valTile, valFloat, tileBase, tileLen, padParams);

            // 与 AccumulateRowScalar 同一套深预取：prologue 发 bDepth-1 个，之后每
            // 消费一个补发一个。只在批内预取，跨批边界让下一批自己的 prologue 发起；
            // NZ_TILE 远大于常见行度数，这个气泡每 256 个非零元最多一次。
            uint32_t jIssue = 0;
            for (uint32_t k = 1; k < bDepth && jIssue < tileLen; k++, jIssue++) {
                IssueBRowFromTile(idxTile, jIssue, colStart, copyParams, padParams);
            }

            for (uint32_t j = 0; j < tileLen; j++) {
                LocalTensor<ValT> bCur = bQueue.DeQue<ValT>();
                float val;
                if constexpr (kIsFloat) {
                    val = valTile.GetValue(j) * alphaVal;
                } else {
                    val = valFloat.GetValue(j) * alphaVal;
                }

                if (jIssue < tileLen) {
                    IssueBRowFromTile(idxTile, jIssue, colStart, copyParams, padParams);
                    jIssue++;
                }

                AxpyBRow(accLocal, tmp, bCur, val, chunkAlign);
            }

            // S 读完两块 tile -> 下一批 MTE2 覆写它们（WAR）。窄类型的 valFloat 由
            // 下一批的 Cast 覆写，而那次 Cast 排在下一批 DMA 之后，所以这一个事件
            // 传递性地也把它挡住了。
            if (tileBase + SPMM_ARCH22_NZ_TILE < re) {
                auto evtWar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
                SetFlag<HardEvent::S_MTE2>(evtWar);
                WaitFlag<HardEvent::S_MTE2>(evtWar);
            }
        }
    }

    // acc += beta * C[row, chunk]。C 借 bQueue 的 tile 搬入，用完立刻归还。
    __aicore__ inline void AccumulateBeta(LocalTensor<float> &accLocal, LocalTensor<float> &tmp,
        uint64_t outBase, uint32_t chunkAlign, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        LocalTensor<ValT> cLocal = bQueue.AllocTensor<ValT>();
        DataCopyPad(cLocal, cGm[outBase], copyParams, padParams);
        bQueue.EnQue(cLocal);
        cLocal = bQueue.DeQue<ValT>();
        if constexpr (kIsFloat) {
            Muls(tmp, cLocal, betaVal, chunkAlign);
        } else {
            LocalTensor<float> cFloat = bFloatBuf.Get<float>();
            Cast<float, ValT>(cFloat, cLocal, RoundMode::CAST_NONE, chunkAlign);
            PipeBarrier<PIPE_V>();
            Muls(tmp, cFloat, betaVal, chunkAlign);
        }
        Add(accLocal, accLocal, tmp, chunkAlign);
        bQueue.FreeTensor(cLocal);
    }

    // 把 fp32 累加器写回 C。ValT != float 时先 Cast 回窄类型再搬出。
    __aicore__ inline void StoreRow(LocalTensor<float> &accLocal, uint64_t outBase,
        uint32_t chunkAlign, DataCopyParams copyParams)
    {
        if constexpr (kIsFloat) {
            outQueue.EnQue(accLocal);
            accLocal = outQueue.DeQue<float>();
            DataCopyPad(yGm[outBase], accLocal, copyParams);
            outQueue.FreeTensor(accLocal);
        } else {
            LocalTensor<ValT> yLocal = outQueue.AllocTensor<ValT>();
            Cast<ValT, float>(yLocal, accLocal, RoundMode::CAST_ROUND, chunkAlign);
            PipeBarrier<PIPE_V>();
            outQueue.EnQue(yLocal);
            yLocal = outQueue.DeQue<ValT>();
            DataCopyPad(yGm[outBase], yLocal, copyParams);
            outQueue.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void SyncScalarToVector();
    __aicore__ inline void SyncVectorToScalar();
    __aicore__ inline void SyncScalarToMte3();
    __aicore__ inline void SyncVectorToMte3();
    __aicore__ inline void SyncMte3ToScalar();
    __aicore__ inline float LoadGeneric(GlobalTensor<ValT> &gm, uint64_t index);
    __aicore__ inline void StoreGeneric(uint64_t index, float value);
    __aicore__ inline uint64_t GenericBIndex(uint32_t k, uint32_t col) const;
    __aicore__ inline uint64_t GenericCIndex(uint32_t row, uint32_t col) const;
    __aicore__ inline void InitGenericLocalBuffers();
    __aicore__ inline void AddGenericProduct(float &acc, float &comp, int &scale, float product);
    __aicore__ inline void AddGenericProductScaled(
        float &acc, float &comp, int &scale, float value, float bValue);
    __aicore__ inline float AccumulateGenericRow(int32_t row, uint32_t col, float &comp, int &scale);
    __aicore__ inline float AccumulateGenericColumn(int32_t row, uint32_t col, float &comp, int &scale);
    __aicore__ inline void ApplyBetaGeneric(
        uint64_t out, float &acc, float &betaProduct, float &betaError);
    __aicore__ inline void ProcessGeneric();

    __aicore__ inline void ProcessChunk(
        uint32_t colStart, uint32_t chunkLen, uint32_t chunkAlign)
    {
        if (rowGroup > 0) {
            ProcessChunkGrouped(colStart, chunkLen, chunkAlign);
            return;
        }
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        DataCopyParams copyParams = {1, (uint16_t)(chunkLen * sizeof(ValT)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        for (uint32_t ri = myRowStart; ri < myRowEnd; ri++) {
            int32_t row = reorderGm.GetValue(ri);
            uint32_t rs = static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseOffset);
            uint32_t re = static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseOffset);
            uint64_t outBase = static_cast<uint64_t>(row) * N + colStart;

            LocalTensor<float> accLocal;
            if constexpr (kIsFloat) {
                accLocal = outQueue.AllocTensor<float>();
            } else {
                accLocal = accBuf.Get<float>();
            }
            Duplicate(accLocal, 0.0f, chunkAlign);

            if (rs < re) {
                AccumulateRow(accLocal, tmp, rs, re, colStart, chunkAlign, copyParams, padParams);
            }
            if (!skipBeta) {
                AccumulateBeta(accLocal, tmp, outBase, chunkAlign, copyParams, padParams);
            }
            StoreRow(accLocal, outBase, chunkAlign, copyParams);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, SPMM_KERNEL_BQUEUE_MAX> bQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> accBuf;     // only used when ValT != float
    TBuf<TPosition::VECCALC> bFloatBuf;  // only used when ValT != float
    TBuf<TPosition::VECCALC> genericRawBuf;
    TBuf<TPosition::VECCALC> genericFloatBuf;
    GlobalTensor<int32_t> transposeRowOffsetsGm;
    GlobalTensor<int32_t> transposeSourceRowsGm;
    GlobalTensor<int32_t> transposePermutationGm;

    uint32_t K;
    uint32_t bDepth;  // bQueue 实际深度，见 SpmmArch22PrefetchDepth
    static constexpr uint32_t genericScalarElems = 16;
    int32_t ldb;
    int32_t ldc;
    int32_t bOrder;
    int32_t cOrder;
    uint32_t aRows;
    int32_t opA;
    int32_t opB;
    bool genericPath;
    bool highPrecision;
};

#include "spmm_kernel_generic.h"

// ---- per-type __global__ kernel 定义（各自 TU 展开一个实例） ----
// KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)：本 kernel 只用 Vector 引擎
// (DataCopyPad + Duplicate/Axpy/Muls/Add/Cast)，完全不碰 Cube。不加这个标记时
// 编译期会报 "kernel type ... is not marked, auto type derivate may be failed"，
// 且只能按 AI Core 数(本机 20)起 block；标记成 AIV_ONLY 后可以按 host 侧
// GetSpmmBlockDim 通过 GetAivCoreCount() 取得的 Vector Core 数起 block，充分使用
// AIV 并行度。仓库里其他 Vector kernel 也使用 AIV_ONLY 标记。
#define DEFINE_SPMM_KERNEL(ValT)                                                       \
    __global__ __aicore__ void spmm_arch22_##ValT(                                     \
        GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,                                \
        GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tilingGm)                \
    {                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                \
        SpmmArch22Kernel<ValT> op;                                                     \
        op.Init(rowOff, colIdx, values, matB, matC, workspace, tilingGm);              \
        op.Process();                                                                  \
    }                                                                                  \
    __global__ __aicore__ void spmm_arch22_##ValT##_out(                               \
        GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,                                \
        GM_ADDR matB, GM_ADDR matC, GM_ADDR matY, GM_ADDR workspace, GM_ADDR tilingGm) \
    {                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                \
        SpmmArch22Kernel<ValT> op;                                                     \
        op.Init(rowOff, colIdx, values, matB, matC, matY, workspace, tilingGm);        \
        op.Process();                                                                  \
    }

// 分发器需要的入口声明集中在 spmm_kernel_entries.h，不在此处提供 extern 宏。
