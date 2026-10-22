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
#include "kernel_operator.h"
#include "../spmm.h"
#include "spmm_kernel_common.h"
#include "spmm_kernel_entry_macros.h"
#include "spmm_kernel_compensated_math.h"

using namespace AscendC;

namespace {
constexpr uint32_t SPMM_C64_BUFFER_NUM = SPMM_ARCH22_BUFFER_NUM;
}

/**
 * @brief CSR 稀疏矩阵 x 稠密矩阵乘法（SpMM），complex64 专用 kernel。
 *
 * complex64 在 GM 上按 aclsparseComplex{float x; float y;} 交织存储(AoS: re,im,
 * re,im,...)，与 float2 二进制兼容。AscendC 向量引擎没有原生复数运算。
 *
 * 本 kernel 全程在 AoS 交织域上算，不做任何反交织/再交织。核心恒等式：
 * 定义"配对换位取负"算子 J，作用在交织向量上是
 *   J([x0,x1, x2,x3, ...]) = [-x1,x0, -x3,x2, ...]
 * 即把每个复数 (a+bi) 映射成 (-b+ai) —— 正好是"乘以虚数单位 i"。于是对交织
 * 存储的复向量 v 与复标量 s = sRe + i*sIm：
 *   s * v = sRe * v + sIm * J(v)
 * 两项都是"实标量 × 交织向量"，可以直接用 Axpy 在交织域上做，完全不需要把
 * re/im 拆成两个 SoA 缓冲。
 *
 * 更进一步：J 是线性算子，因此可以和求和交换次序。一行的累加
 *   acc = Σ_k s_k * v_k = Σ_k (sRe_k * v_k) + J(Σ_k (sIm_k * v_k))
 * 只要维护两个交织累加器 accA/accB：
 *   accA += sRe_k * v_k     (每个非零元 1 条 Axpy)
 *   accB += sIm_k * v_k     (每个非零元 1 条 Axpy)
 * 行结束后再做一次 acc = accA + J(accB) 即可。也就是说 **J 只在每行末尾算
 * 一次**，而不是每个非零元算一次。
 *
 * 每个非零元的向量指令数因此降到 2 条(2 条 Axpy)。对比历史版本：
 *   - 最早版本：GatherMask×2 反交织 + Muls/Muls/Sub/Add/Muls/Muls/Add/Add
 *     = 10 条/非零元，外加每行一次 O(chunkAlign) 的 Set/GetValue 标量交织循环。
 *   - 上一版(P8)：GatherMask×2 反交织 + Axpy×4 = 6 条/非零元，外加每行一次
 *     Gather 再交织。
 *   - 本版：Axpy×2 = 2 条/非零元，J 每行一次(Gather + Mul)，且累加器本身
 *     就是 AoS 布局，写回 Y 时不需要任何交织操作，直接连续 DataCopyPad。
 *
 * J 的实现：Gather 按预计算的"配对下标表"(偶数位取奇数位、奇数位取偶数位)
 * 做换位，再 Mul 一个固定的 ±1 符号向量(偶数位 -1、奇数位 +1)完成取负。
 * 两张表都只依赖固定的 buffer 宽度，在 Init() 里用标量循环算一次，不在按行
 * 走的热路径上。Gather 是 count 语义的 Level 2 API，内部按 64 元素一组自动
 * 拆 repeat(见 kernel_operator_vec_gather_intf_impl.h 的 count 版本)。
 *
 * 数值精度说明：新写法把实部算成 Σ(sRe*re) - Σ(sIm*im)(两个和分别累加后相减)，
 * 而旧写法是 Σ(sRe*re - sIm*im)(逐项相减后累加)。两者数学等价，浮点下抵消
 * 行为略有差异，因此这版改动用全量 200 case 精度套件 + 高压 stress probe
 * 重新验证过，不只跑小用例。
 */
class SpmmArch22KernelComplex64 : public SpmmArch22KernelCommon {
    // Init()/Process() 入口样板与实数版逐字一致，收敛到共享宏（见
    // spmm_kernel_entry_macros.h）；complex64 非 generic 分支需额外建 J 算子表，
    // 经 POST 形参注入。
    SPMM_ARCH22_DEFINE_KERNEL_ENTRY(InitJOperatorTables();)

private:
    // 基类的列分块遍历需要回调下面 private 的 ProcessChunk。
    friend class SpmmArch22KernelCommon;

    // 与值类型无关的字段由基类的 InitCommonTiling 读取，这里只补复数版特有的
    // alpha/beta 实部虚部。
    __aicore__ inline void InitTiling(__gm__ SpmmArch22TilingData *td)
    {
        InitCommonTiling(td);
        alphaRe = td->alphaHost;
        alphaIm = td->alphaImHost;
        betaRe = td->betaHost;
        betaIm = td->betaImHost;
        ldb = td->ldbHost;
        ldc = td->ldcHost;
        bOrder = td->bOrderHost;
        cOrder = td->cOrderHost;
        aRows = td->aRowsHost;
        opA = td->opAHost;
        opB = td->opBHost;
        genericPath = (td->genericPathHost != 0);
        numericGuard = (td->numericGuardHost != 0);
        // 复数 beta 两个分量都为 0 才算 0，与 torch.sparse.addmm 的短路语义一致。
        // 这一条同时堵死 P2 的通道：beta==0 时不再把 C 借批槽搬进 UB，即使
        // rowGroup 钳位失效，C 里的 NaN/Inf 也进不了累加器。
        skipBeta = (betaRe == 0.0f && betaIm == 0.0f);
    }

    __aicore__ inline uint64_t StorageElems(
        uint32_t rows, uint32_t cols, int32_t ld, int32_t order)
    {
        if (rows == 0 || cols == 0) {
            return 0;
        }
        return order == 0 ? static_cast<uint64_t>(rows - 1) * ld + cols :
                            static_cast<uint64_t>(cols - 1) * ld + rows;
    }

    __aicore__ inline void InitGlobalBuffers(__gm__ SpmmArch22TilingData *td, GM_ADDR rowOff,
        GM_ADDR colIdx, GM_ADDR values, GM_ADDR matB, GM_ADDR matC, GM_ADDR matY,
        GM_ADDR workspace)
    {
        rowOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowOff), aRows + 1);
        colIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(colIdx), nnz);
        // complex64 = 2 个 float(re,im)，用 float GlobalTensor 承载，元素数乘 2。
        valuesGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(values), static_cast<uint64_t>(nnz) * 2);
        const uint64_t bElems = StorageElems(td->bRowsHost, td->bColsHost, ldb, bOrder);
        const uint64_t cElems = StorageElems(M, N, ldc, cOrder);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matB), bElems * 2);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matC), cElems * 2);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matY), cElems * 2);
        SPMM_ARCH22_BIND_TRANSPOSE_GM(td, workspace);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        // UB 预算的算式在 spmm.h 里，host 的 chunk 划分与这里的缓冲宽度必须同源，
        // 否则 chunk 边界会错位。
        nMaxPerChunk = SpmmArch22NMaxChunkComplex64(N);
        nAlignChunk = ROUNDUP(nMaxPerChunk, SPMM_ARCH22_ALIGN);

        uint32_t aosBytes = static_cast<uint32_t>(nAlignChunk * 2 * sizeof(float));
        pipe.InitBuffer(rawQueue, SPMM_C64_BUFFER_NUM, aosBytes);
        pipe.InitBuffer(outQueue, 1, aosBytes);
        pipe.InitBuffer(accABuf, aosBytes);
        pipe.InitBuffer(accBBuf, aosBytes);
        pipe.InitBuffer(swappedBuf, aosBytes);
        pipe.InitBuffer(signBuf, aosBytes);
        pipe.InitBuffer(pairOffsetBuf, nAlignChunk * 2 * sizeof(uint32_t));

        // AccumulateRowBatched 的槽位数组，用 nMax 定下来之后 UB 里剩的余量分配，
        // 与 kernels/spmm_kernel.h 里实数版同一套口径。这不影响 nMaxPerChunk，
        // 所以 host 侧 SpmmArch22ComputeNChunksComplex64 不必跟着改。
        const uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
        const uint64_t used =
            SPMM_ARCH22_C64_FLOATS_PER_ELEM * nAlignChunk * sizeof(float);
        batchSlots = 0;
        for (uint32_t s = SPMM_ARCH22_BATCH_SLOTS_MAX; s >= SPMM_ARCH22_BATCH_SLOTS_MIN;
             s >>= 1) {
            if (used + static_cast<uint64_t>(s) * aosBytes <= availUB) {
                batchSlots = s;
                break;
            }
        }
        if (batchSlots > 0) {
            pipe.InitBuffer(bBatchBuf, batchSlots * aosBytes);
        }

        // 行分组：把每行的固定开销（reorder/rowOffsets 的散读、C 的两次 DMA、
        // outQueue 的 TQue 同步）摊到一组。实数版实测这笔开销在小行度数下占到
        // 84%，改成分组 + host 侧连续行块装箱后 P-02 提升 4.1~4.6 倍。complex64
        // 的行结构与实数版一致（只是每行宽一倍、行末多一个 J 算子），同一改法适用。
        // 粒度与 host 侧 DoPreProcess 取自同一个函数。
        rowGroup = SpmmArch22RowGroupComplex64(N);
        // 兜底：InitGroupAcc 把整组的 C 借批槽搬入，组内行数超过槽位数就会写出
        // bBatchBuf。上面那个函数已经把 g 钳到 slots 以内，这里再挡一次，避免将来
        // 两个上限常量各自变动时又从这里漏出去。等值时不影响 host 的装箱粒度。
        if (rowGroup > batchSlots) {
            rowGroup = batchSlots;
        }
        if (rowGroup > 0) {
            pipe.InitBuffer(accAGroupBuf, rowGroup * aosBytes);
            pipe.InitBuffer(accBGroupBuf, rowGroup * aosBytes);
            pipe.InitBuffer(rowOffBuf, (SPMM_ARCH22_ROW_GROUP_MAX + 1) * sizeof(int32_t));
            fuseGroupDma = (nAlignChunk == N);
        } else {
            fuseGroupDma = false;
        }
        // 快路径的数值安全回退复用 StoreGeneric 的 8-float 暂存；定长、不进按 N
        // 伸缩的 UB 预算。
        pipe.InitBuffer(genericOutBuf, 8 * sizeof(float));
    }

    // J 算子的两张固定表：pairOffset 把下标 j 映射到 j^1(偶<->奇 换位)的字节
    // 偏移，sign 在偶数位放 -1、奇数位放 +1。只依赖 buffer 宽度，与每个 chunk
    // 实际用到多少列(chunkLen/chunkAlign 会变)无关，Init() 里算一次即可。
    __aicore__ inline void InitJOperatorTables()
    {
        LocalTensor<uint32_t> pairOffset = pairOffsetBuf.Get<uint32_t>();
        LocalTensor<float> sign = signBuf.Get<float>();
        uint32_t aosLen = nAlignChunk * 2;
        for (uint32_t j = 0; j < aosLen; j++) {
            pairOffset.SetValue(j, static_cast<uint32_t>((j ^ 1U) * sizeof(float)));
            sign.SetValue(j, ((j & 1U) == 0U) ? -1.0f : 1.0f);
        }
        // 这两张表是标量(PIPE_S)写进 UB 的，而读它们的是 StoreRow 里的
        // Gather/Mul(PIPE_V)。pairOffsetBuf/signBuf 是裸 TBuf，没有 TQue 那种
        // EnQue/DeQue 依赖跟踪，所以标量写和向量读之间必须显式同步，否则
        // 是一处 S->V 竞态(实测因指令间距足够大而未复现，但依赖运气)。
        // 写法与仓库内 coosort/arch35 一致。
        auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evt);
        WaitFlag<HardEvent::S_V>(evt);
    }
    // 累加一行的全部非零元。s = alpha*A[row,k] 先折成一对实标量(sRe,sIm)，交织
    // 域上就只剩两条实数 Axpy：accA += sRe*raw、accB += sIm*raw。复数乘里那个
    // "换位取负"留到行末由 J 算子一次性折算(见 StoreRow)，靠的是 J 的线性性。
    __aicore__ inline void AccumulateRow(LocalTensor<float> &accA, LocalTensor<float> &accB,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t aosLen,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t col0 = static_cast<uint32_t>(colIndicesGm.GetValue(rs) - idxBaseOffset);
        LocalTensor<float> rawFirst = rawQueue.AllocTensor<float>();
        DataCopyPad(rawFirst, bGm[(static_cast<uint64_t>(col0) * N + colStart) * 2],
                    copyParams, padParams);
        rawQueue.EnQue(rawFirst);

        for (uint32_t idx = rs; idx < re; idx++) {
            LocalTensor<float> raw = rawQueue.DeQue<float>();

            float vRe = valuesGm.GetValue(static_cast<uint64_t>(idx) * 2);
            float vIm = valuesGm.GetValue(static_cast<uint64_t>(idx) * 2 + 1);
            // alpha 折进每个非零元的标量里，不额外占一遍向量指令。
            float sRe;
            float sIm;
            ScaleAlphaInto(vRe, vIm, sRe, sIm);

            if (idx + 1 < re) {
                uint32_t nextCol = static_cast<uint32_t>(colIndicesGm.GetValue(idx + 1) - idxBaseOffset);
                LocalTensor<float> rawNext = rawQueue.AllocTensor<float>();
                DataCopyPad(rawNext, bGm[(static_cast<uint64_t>(nextCol) * N + colStart) * 2],
                            copyParams, padParams);
                rawQueue.EnQue(rawNext);
            }

            // 每个非零元只有这 2 条向量指令：交织域上的实标量 Axpy。
            Axpy(accA, raw, sRe, static_cast<int32_t>(aosLen));
            Axpy(accB, raw, sIm, static_cast<int32_t>(aosLen));

            rawQueue.FreeTensor(raw);
        }
    }

    // 整批版：一批 batchSlots 个非零元共用一组同步事件，语义与 AccumulateRow 完全
    // 相同（同样是每非零元两条交织域 Axpy，J 算子仍留到行末）。
    //
    // 差别只在不再走 TQue：那条路径每个非零元都要 AllocTensor/EnQue/DeQue/
    // FreeTensor，每次都带跨引擎事件。实数版实测这笔逐元素簿记就是小 N 的主要开销，
    // 改成整批后 P-02 fp32 1.54x、fp16 1.72x、P-03 fp32 2.22x；complex64 的 B 行
    // gather 结构与实数版一致（只是每行宽一倍），所以同一改法直接适用。
    __aicore__ inline void AccumulateRowBatched(LocalTensor<float> &accA, LocalTensor<float> &accB,
        uint32_t rs, uint32_t re, uint32_t colStart, uint32_t aosLen,
        DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        LocalTensor<float> batch = bBatchBuf.Get<float>();
        const uint32_t slotStride = nAlignChunk * 2;

        for (uint32_t base = rs; base < re; base += batchSlots) {
            uint32_t cnt = re - base;
            if (cnt > batchSlots) {
                cnt = batchSlots;
            }

            for (uint32_t j = 0; j < cnt; j++) {
                uint32_t col =
                    static_cast<uint32_t>(colIndicesGm.GetValue(base + j) - idxBaseOffset);
                DataCopyPad(batch[j * slotStride],
                    bGm[(static_cast<uint64_t>(col) * N + colStart) * 2], copyParams, padParams);
            }

            auto evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evIn);
            WaitFlag<HardEvent::MTE2_V>(evIn);

            for (uint32_t j = 0; j < cnt; j++) {
                uint64_t vi = static_cast<uint64_t>(base + j) * 2;
                float vRe = valuesGm.GetValue(vi);
                float vIm = valuesGm.GetValue(vi + 1);
                float sRe;
                float sIm;
                ScaleAlphaInto(vRe, vIm, sRe, sIm);

                LocalTensor<float> slot = batch[j * slotStride];
                Axpy(accA, slot, sRe, static_cast<int32_t>(aosLen));
                Axpy(accB, slot, sIm, static_cast<int32_t>(aosLen));
            }

            // WAR：复写槽位前要等本批的 Axpy(PIPE_V) 读完。同实数内核，必须无条件
            // 发——StoreRow 里 outQueue 的 EnQue/DeQue 只挡 MTE3，挡不住下一行填槽的
            // MTE2，有条件发会在 rowGroup==0 而 batchSlots>0 的 N 区间跨行覆写。
            auto evWar =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evWar);
            WaitFlag<HardEvent::V_MTE2>(evWar);
        }
    }

    // beta*C 走完全相同的分解：accA += betaRe*C、accB += betaIm*C，由行末统一的
    // accA + J(accB) 一起折算，这里不需要单独的复数乘法。
    __aicore__ inline void AccumulateBeta(LocalTensor<float> &accA, LocalTensor<float> &accB,
        uint64_t outBase, uint32_t aosLen, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        LocalTensor<float> rawC = rawQueue.AllocTensor<float>();
        DataCopyPad(rawC, cGm[outBase], copyParams, padParams);
        rawQueue.EnQue(rawC);
        rawC = rawQueue.DeQue<float>();

        Axpy(accA, rawC, betaRe, static_cast<int32_t>(aosLen));
        Axpy(accB, rawC, betaIm, static_cast<int32_t>(aosLen));

        rawQueue.FreeTensor(rawC);
    }

    // 判非有限：NaN 判据 v != v，±Inf 判据 (v - v) 得 NaN。避免 union 类型双关
    // （C++ 标准里读写 union 不同成员是未定义行为）。
    __aicore__ inline bool IsFiniteF(float v)
    {
        return (v == v) && ((v - v) == 0.0f);
    }

    // 提前合并 alpha：s = alpha*A。若 alpha*A 溢出（非有限）或某分量的非零贡献项
    // 下溢成 0，标记 fastAlphaRangeLost，行末回退到 generic 的 ScaledComplexProduct
    // 做范围保护。按"贡献项非零"判断，避免把精确零分量（如纯实输入的 sIm==0）误判
    // 为下溢；只判 sRe/sIm 全零又会漏掉"单分量下溢"。
    __aicore__ inline void ScaleAlphaInto(float vRe, float vIm, float &sRe, float &sIm)
    {
        sRe = alphaRe * vRe - alphaIm * vIm;
        sIm = alphaRe * vIm + alphaIm * vRe;
        const bool sReLost = (sRe == 0.0f) && ((alphaRe != 0.0f && vRe != 0.0f) ||
                                               (alphaIm != 0.0f && vIm != 0.0f));
        const bool sImLost = (sIm == 0.0f) && ((alphaRe != 0.0f && vIm != 0.0f) ||
                                               (alphaIm != 0.0f && vRe != 0.0f));
        if (sReLost || sImLost || !IsFiniteF(sRe) || !IsFiniteF(sIm)) {
            fastAlphaRangeLost = true;
        }
    }

    // 检测 AoS 累加器 [0, aosLen) 是否含非有限值。先 V->S 同步一次，让合成 Add 的
    // 向量写对后续标量读可见（与 generic 里 LoadGeneric 的既有模式一致）。
    __aicore__ inline bool AccNonFinite(LocalTensor<float> acc, uint32_t aosLen)
    {
        auto evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evt);
        WaitFlag<HardEvent::V_S>(evt);
        for (uint32_t i = 0; i < aosLen; i++) {
            if (!IsFiniteF(acc.GetValue(i))) {
                return true;
            }
        }
        return false;
    }

    // 对 row 在 [colStart, colEnd) 列区间用 generic 逐元素重算并写回。快路径仅在
    // opA/opB 非转置、B/C 行主序、ld==N 时启用（host 侧 SpmmUsesFastPath 保证），
    // 故 GenericCIndex 与快路径 outBase 逐列对齐，可直接复用。
    __aicore__ inline void FallbackGenericRowChunk(int32_t row, uint32_t colStart, uint32_t colEnd)
    {
        for (uint32_t col = colStart; col < colEnd; col++) {
            float accRe = 0.0f;
            float accIm = 0.0f;
            AccumulateGenericRow(row, col, accRe, accIm);
            const uint64_t out = GenericCIndex(static_cast<uint32_t>(row), col);
            AddGenericBeta(out, accRe, accIm);
            StoreGeneric(out, accRe, accIm);
        }
    }

    // 行末一次性算 J(accB) 并合并：swapped = J(accB)，out = accA + swapped。
    // 最后这条 Add 直接把结果写进 outQueue 的 buffer，顺带完成"搬进 DMA 输出
    // 缓冲"这一步，不额外多一条拷贝指令；输出路径仍走已验证的 TQue
    // EnQue/DeQue 同步。累加器本身就是 AoS，写回不需要交织。
    __aicore__ inline void StoreRow(LocalTensor<float> &accA, LocalTensor<float> &accB,
        LocalTensor<float> &swapped, LocalTensor<float> &sign,
        LocalTensor<uint32_t> &pairOffset, uint64_t outBase, uint32_t aosLen,
        DataCopyParams copyParams, int32_t row, uint32_t colStart, uint32_t chunkLen)
    {
        LocalTensor<float> outRaw = outQueue.AllocTensor<float>();
        PipeBarrier<PIPE_V>();
        Gather(swapped, accB, pairOffset, (uint32_t)0, aosLen);
        PipeBarrier<PIPE_V>();
        Mul(swapped, swapped, sign, aosLen);
        PipeBarrier<PIPE_V>();
        Add(outRaw, accA, swapped, aosLen);
        PipeBarrier<PIPE_V>();

        // 合成后、写回前检测；TQue 流程始终平衡（坏行只跳过 DataCopyPad）。
        // fastAlphaRangeLost 覆盖提前合并 alpha 的下溢/溢出（非有限之外的"错误有限零"）。
        const bool bad = (numericGuard && AccNonFinite(outRaw, aosLen)) || fastAlphaRangeLost;

        outQueue.EnQue(outRaw);
        outRaw = outQueue.DeQue<float>();
        if (!bad) {
            DataCopyPad(yGm[outBase], outRaw, copyParams);
        }
        outQueue.FreeTensor(outRaw);

        if (bad) {
            FallbackGenericRowChunk(row, colStart, colStart + chunkLen);
        }
    }

    // 一组行的共享标量，与实数版 GroupCtx 同构，多出 complex64 的 AoS 长度与步长。
    struct GroupCtx {
        uint32_t gCnt;
        uint32_t colStart;
        uint32_t chunkLen;
        uint32_t aosLen;     // chunkAlign * 2，一行实虚交错后的元素数
        uint32_t stride;     // nAlignChunk * 2，组内各行在 UB 里的步长
        int32_t row0;
        bool contig;
        bool fused;
        uint64_t groupBase;
    };

    // 两个累加器清零，beta!=0 时再各自叠上 beta*C。
    __aicore__ inline void InitGroupAcc(LocalTensor<float> &accA, LocalTensor<float> &accB,
        LocalTensor<float> &batch, const GroupCtx &gc, DataCopyParams copyParams,
        DataCopyPadParams padParams)
    {
        Duplicate(accA, 0.0f, gc.gCnt * gc.stride);
        Duplicate(accB, 0.0f, gc.gCnt * gc.stride);
        if (skipBeta) {
            return;
        }
        // C 借批槽搬入：此刻批槽还没开始用，省一块与分组同宽的 UB。
        // 槽位数不少于 rowGroup（两者取自同一份 UB 余量，槽位先分），够放。
        if (gc.fused) {
            DataCopyParams groupCopy{static_cast<uint16_t>(gc.gCnt),
                static_cast<uint16_t>(gc.chunkLen * 2 * sizeof(float)), 0, 0};
            DataCopyPad(batch, cGm[gc.groupBase], groupCopy, padParams);
        } else {
            for (uint32_t g = 0; g < gc.gCnt; g++) {
                DataCopyPad(batch[g * gc.stride],
                    cGm[(static_cast<uint64_t>(grpRowIds[g]) * N + gc.colStart) * 2],
                    copyParams, padParams);
            }
        }
        auto evC = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evC);
        WaitFlag<HardEvent::MTE2_V>(evC);
        for (uint32_t g = 0; g < gc.gCnt; g++) {
            Axpy(accA[g * gc.stride], batch[g * gc.stride], betaRe,
                static_cast<int32_t>(gc.aosLen));
            Axpy(accB[g * gc.stride], batch[g * gc.stride], betaIm,
                static_cast<int32_t>(gc.aosLen));
        }
        // WAR：后面填批槽的 MTE2 要等上面这些 Axpy 读完批槽。
        auto evW = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evW);
        WaitFlag<HardEvent::V_MTE2>(evW);
    }

    // 行末的 J 算子与合并。换位表只有一行宽，用 Gather 的 srcBaseAddr 定位到组内
    // 第 g 行，结果就地加回 accA，写回时直接搬 accA，不需要额外输出缓冲。
    __aicore__ inline uint32_t ApplyJOperator(LocalTensor<float> &accA, LocalTensor<float> &accB,
        LocalTensor<float> &swapped, LocalTensor<float> &sign,
        LocalTensor<uint32_t> &pairOffset, const GroupCtx &gc)
    {
        uint32_t bad = 0;
        PipeBarrier<PIPE_V>();
        for (uint32_t g = 0; g < gc.gCnt; g++) {
            Gather(swapped, accB, pairOffset,
                static_cast<uint32_t>(g * gc.stride * sizeof(float)), gc.aosLen);
            PipeBarrier<PIPE_V>();
            Mul(swapped, swapped, sign, gc.aosLen);
            PipeBarrier<PIPE_V>();
            Add(accA[g * gc.stride], accA[g * gc.stride], swapped, gc.aosLen);
            PipeBarrier<PIPE_V>();
            if ((numericGuard && AccNonFinite(accA[g * gc.stride], gc.aosLen)) || fastAlphaRangeLost) {
                bad |= (1u << (g & 31u));
            }
        }
        return bad;
    }

    __aicore__ inline void StoreGroupAcc(LocalTensor<float> &accA, const GroupCtx &gc,
        DataCopyParams copyParams, uint32_t badMask)
    {
        auto evOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evOut);
        WaitFlag<HardEvent::V_MTE3>(evOut);
        if (gc.fused && badMask == 0) {
            DataCopyParams groupCopy{static_cast<uint16_t>(gc.gCnt),
                static_cast<uint16_t>(gc.chunkLen * 2 * sizeof(float)), 0, 0};
            DataCopyPad(yGm[gc.groupBase], accA, groupCopy);
            return;
        }
        for (uint32_t g = 0; g < gc.gCnt; g++) {
            if (badMask & (1u << (g & 31u))) {
                continue;
            }
            DataCopyPad(yGm[(static_cast<uint64_t>(grpRowIds[g]) * N + gc.colStart) * 2],
                accA[g * gc.stride], copyParams);
        }
    }

    // 一组 gCnt 行共用一遍固定开销。与实数版 ProcessChunkGrouped 同构，差别只在
    // 每行有两个累加器、且行末要各做一次 J 算子（换位表只有一行宽，所以 J 逐行做；
    // 它是三条纯向量指令，没有跨引擎同步，不是这里要省的东西）。
    __aicore__ inline void ProcessChunkGrouped(
        uint32_t colStart, uint32_t chunkLen, uint32_t chunkAlign)
    {
        DataCopyParams copyParams = {1, (uint16_t)(chunkLen * 2 * sizeof(float)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        const uint32_t aosLen = chunkAlign * 2;
        // 组内各行在 UB 里的步长按 nAlignChunk 取，不随末尾短 chunk 变化。
        const uint32_t stride = nAlignChunk * 2;

        LocalTensor<float> accA = accAGroupBuf.Get<float>();
        LocalTensor<float> accB = accBGroupBuf.Get<float>();
        LocalTensor<float> swapped = swappedBuf.Get<float>();
        LocalTensor<float> sign = signBuf.Get<float>();
        LocalTensor<uint32_t> pairOffset = pairOffsetBuf.Get<uint32_t>();
        LocalTensor<float> batch = bBatchBuf.Get<float>();

        for (uint32_t gBase = myRowStart; gBase < myRowEnd; gBase += rowGroup) {
            GroupCtx gc;
            gc.gCnt = myRowEnd - gBase;
            if (gc.gCnt > rowGroup) {
                gc.gCnt = rowGroup;
            }
            gc.colStart = colStart;
            gc.chunkLen = chunkLen;
            gc.aosLen = aosLen;
            gc.stride = stride;
            gc.contig = LoadGroupRows(gBase, gc.gCnt, gc.row0, padParams);
            gc.fused = gc.contig && fuseGroupDma;
            gc.groupBase = (static_cast<uint64_t>(gc.row0) * N + colStart) * 2;

            InitGroupAcc(accA, accB, batch, gc, copyParams, padParams);
            fastAlphaRangeLost = false;
            AccumulateGroup(accA, accB, batch, grpRs, grpRe, gc.gCnt, colStart, aosLen, stride,
                copyParams, padParams);
            const uint32_t bad = ApplyJOperator(accA, accB, swapped, sign, pairOffset, gc);
            StoreGroupAcc(accA, gc, copyParams, bad);
            for (uint32_t g = 0; g < gc.gCnt; g++) {
                if (bad & (1u << (g & 31u))) {
                    FallbackGenericRowChunk(static_cast<int32_t>(grpRowIds[g]),
                        gc.colStart, gc.colStart + gc.chunkLen);
                }
            }

            // WAR：下一组的 Duplicate(PIPE_V) 会重写 accA/accB，要等本组 MTE3 读完。
            if (gBase + rowGroup < myRowEnd) {
                auto evW1 =
                    static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(evW1);
                WaitFlag<HardEvent::MTE3_V>(evW1);
            }
        }
    }

    // 填一批 B 行槽位：从 (g, idx) 起沿组内各行推进，最多填 batchSlots 个。
    // 返回实际填入个数，并把游标推进到下一批的起点。
    __aicore__ inline uint32_t FillGroupBatch(LocalTensor<float> &batch, const uint32_t *rsArr,
        const uint32_t *reArr, uint32_t gCnt, uint32_t &g, uint32_t &idx, uint32_t colStart,
        uint32_t stride, DataCopyParams copyParams, DataCopyPadParams padParams)
    {
        uint32_t cnt = 0;
        while (cnt < batchSlots && AdvanceGroupCursor(rsArr, reArr, gCnt, g, idx)) {
            const uint32_t col =
                static_cast<uint32_t>(colIndicesGm.GetValue(idx) - idxBaseOffset);
            DataCopyPad(batch[cnt * stride],
                bGm[(static_cast<uint64_t>(col) * N + colStart) * 2], copyParams, padParams);
            cnt++;
            idx++;
        }
        return cnt;
    }

    // 把一批槽位累加进各自所属行的两个累加器，用 (g0, idx0) 重放填槽游标来定位。
    __aicore__ inline void AxpyGroupBatch(LocalTensor<float> &accA, LocalTensor<float> &accB,
        LocalTensor<float> &batch, const uint32_t *rsArr, const uint32_t *reArr, uint32_t gCnt,
        uint32_t g0, uint32_t idx0, uint32_t cnt, uint32_t aosLen, uint32_t stride)
    {
        uint32_t gg = g0;
        uint32_t ii = idx0;
        for (uint32_t j = 0; j < cnt; j++) {
            while (gg < gCnt && ii >= reArr[gg]) {
                gg++;
                if (gg < gCnt) {
                    ii = rsArr[gg];
                }
            }
            const uint64_t vi = static_cast<uint64_t>(ii) * 2;
            const float vRe = valuesGm.GetValue(vi);
            const float vIm = valuesGm.GetValue(vi + 1);
            float sRe;
            float sIm;
            ScaleAlphaInto(vRe, vIm, sRe, sIm);
            LocalTensor<float> slot = batch[j * stride];
            Axpy(accA[gg * stride], slot, sRe, static_cast<int32_t>(aosLen));
            Axpy(accB[gg * stride], slot, sIm, static_cast<int32_t>(aosLen));
            ii++;
        }
    }

    // 一组内所有行的非零元，按 batchSlots 个一批处理，批可跨行边界。槽位到累加器的
    // 对应关系靠重放游标推导，不另存数组，与实数版 AccumulateGroup 同构。
    __aicore__ inline void AccumulateGroup(LocalTensor<float> &accA, LocalTensor<float> &accB,
        LocalTensor<float> &batch, uint32_t *rsArr, uint32_t *reArr, uint32_t gCnt,
        uint32_t colStart, uint32_t aosLen, uint32_t stride, DataCopyParams copyParams,
        DataCopyPadParams padParams)
    {
        uint32_t g = 0;
        uint32_t idx = rsArr[0];
        while (g < gCnt) {
            const uint32_t g0 = g;
            const uint32_t idx0 = idx;
            const uint32_t cnt = FillGroupBatch(batch, rsArr, reArr, gCnt, g, idx, colStart,
                stride, copyParams, padParams);
            if (cnt == 0) {
                break;
            }

            auto evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evIn);
            WaitFlag<HardEvent::MTE2_V>(evIn);

            AxpyGroupBatch(accA, accB, batch, rsArr, reArr, gCnt, g0, idx0, cnt, aosLen, stride);

            // WAR：复写槽位前等本批 Axpy 读完。无条件发（含本组最后一批）——写回前的
            // V_MTE3 只挡 MTE3，挡不住下一组填槽的 MTE2；分组后它摊到每 rowGroup 行
            // 一次，代价可以忽略。
            auto evWar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(evWar);
            WaitFlag<HardEvent::V_MTE2>(evWar);
        }
    }

    __aicore__ inline uint64_t GenericBIndex(uint32_t k, uint32_t col) const
    {
        const uint32_t row = opB == 0 ? k : col;
        const uint32_t column = opB == 0 ? col : k;
        return bOrder == 0 ? static_cast<uint64_t>(row) * ldb + column :
                             static_cast<uint64_t>(column) * ldb + row;
    }

    __aicore__ inline uint64_t GenericCIndex(uint32_t row, uint32_t col) const
    {
        return cOrder == 0 ? static_cast<uint64_t>(row) * ldc + col :
                             static_cast<uint64_t>(col) * ldc + row;
    }

    struct ScaledReal {
        float value;
        int exponent;
        float low;
    };

    __aicore__ inline ScaledReal NormalizeScaled(float value, int exponent = 0, float low = 0.0f)
    {
        const float combined = value + low;
        const float recovered = combined - value;
        low = (value - (combined - recovered)) + (low - recovered);
        value = combined;
        constexpr float up = 4294967296.0f;
        constexpr float down = 0.00000000023283064365386962890625f;
        float magnitude = value < 0.0f ? -value : value;
        while (magnitude >= up) {
            value *= down;
            low *= down;
            magnitude *= down;
            exponent += 32;
        }
        while (magnitude != 0.0f && magnitude < down) {
            value *= up;
            low *= up;
            magnitude *= up;
            exponent -= 32;
        }
        while (magnitude >= 1.0f) {
            value *= 0.5f;
            low *= 0.5f;
            magnitude *= 0.5f;
            exponent++;
        }
        while (magnitude != 0.0f && magnitude < 0.5f) {
            value *= 2.0f;
            low *= 2.0f;
            magnitude *= 2.0f;
            exponent--;
        }
        return {value, exponent, low};
    }

    __aicore__ inline float RestoreScaled(float value, int exponent)
    {
        while (exponent >= 32) {
            value *= 4294967296.0f;
            exponent -= 32;
        }
        while (exponent <= -32) {
            value *= 0.00000000023283064365386962890625f;
            exponent += 32;
        }
        while (exponent > 0) {
            value *= 2.0f;
            exponent--;
        }
        while (exponent < 0) {
            value *= 0.5f;
            exponent++;
        }
        return value;
    }

    __aicore__ inline ScaledReal TripleProduct(ScaledReal a, ScaledReal b, ScaledReal c)
    {
        const volatile float first = a.value * b.value;
        const float firstError = SpmmProductResidual(a.value, b.value, first);
        const volatile float product = first * c.value;
        const float error = SpmmProductResidual(first, c.value, product) + firstError * c.value;
        return NormalizeScaled(product, a.exponent + b.exponent + c.exponent, error);
    }

    __aicore__ inline float SumScaledProducts(ScaledReal (&terms)[4])
    {
        // Cancel large terms before restoring the FP32 exponent or adding small terms.
        for (uint32_t i = 1; i < 4; i++) {
            for (uint32_t j = i; j > 0 && terms[j].exponent > terms[j - 1].exponent; j--) {
                const ScaledReal previous = terms[j - 1];
                terms[j - 1] = terms[j];
                terms[j] = previous;
            }
        }
        ScaledReal sum = {0.0f, 0, 0.0f};
        for (uint32_t i = 0; i < 4; i++) {
            if (sum.value == 0.0f) {
                sum = terms[i];
            } else if (terms[i].value != 0.0f) {
                const int exponent = sum.exponent > terms[i].exponent ? sum.exponent : terms[i].exponent;
                const float left = RestoreScaled(sum.value, sum.exponent - exponent);
                const float right = RestoreScaled(terms[i].value, terms[i].exponent - exponent);
                const float combined = left + right;
                const float recovered = combined - left;
                const float error = (left - (combined - recovered)) + (right - recovered);
                const float low = RestoreScaled(sum.low, sum.exponent - exponent) +
                    RestoreScaled(terms[i].low, terms[i].exponent - exponent);
                sum = NormalizeScaled(combined, exponent, error + low);
            }
        }
        return RestoreScaled(sum.value + sum.low, sum.exponent);
    }

    __aicore__ inline void ScaledComplexProduct(
        float aRe, float aIm, float bRe, float bIm, float &productRe, float &productIm)
    {
        const ScaledReal ar = NormalizeScaled(alphaRe);
        const ScaledReal ai = NormalizeScaled(alphaIm);
        const ScaledReal xr = NormalizeScaled(aRe);
        const ScaledReal xi = NormalizeScaled(aIm);
        const ScaledReal yr = NormalizeScaled(bRe);
        const ScaledReal yi = NormalizeScaled(bIm);
        ScaledReal realTerms[4] = {
            TripleProduct(ar, xr, yr), TripleProduct(ai, xi, yr),
            TripleProduct(ar, xi, yi), TripleProduct(ai, xr, yi)};
        ScaledReal imagTerms[4] = {
            TripleProduct(ar, xr, yi), TripleProduct(ai, xi, yi),
            TripleProduct(ar, xi, yr), TripleProduct(ai, xr, yr)};
        realTerms[1].value = -realTerms[1].value;
        realTerms[1].low = -realTerms[1].low;
        realTerms[2].value = -realTerms[2].value;
        realTerms[2].low = -realTerms[2].low;
        realTerms[3].value = -realTerms[3].value;
        realTerms[3].low = -realTerms[3].low;
        imagTerms[1].value = -imagTerms[1].value;
        imagTerms[1].low = -imagTerms[1].low;
        productRe = SumScaledProducts(realTerms);
        productIm = SumScaledProducts(imagTerms);
    }

    __aicore__ inline void AccumulateGenericProduct(
        uint32_t idx, uint32_t k, uint32_t col, float &accRe, float &accIm)
    {
        const uint64_t vi = static_cast<uint64_t>(idx) * 2;
        float aRe = valuesGm.GetValue(vi);
        float aIm = valuesGm.GetValue(vi + 1);
        if (opA == 2) {
            aIm = -aIm;
        }
        const uint64_t bi = GenericBIndex(k, col) * 2;
        const float bRe = bGm.GetValue(bi);
        float bIm = bGm.GetValue(bi + 1);
        if (opB == 2) {
            bIm = -bIm;
        }
        float productRe;
        float productIm;
        if (IsFiniteF(alphaRe) && IsFiniteF(alphaIm) && IsFiniteF(aRe) &&
            IsFiniteF(aIm) && IsFiniteF(bRe) && IsFiniteF(bIm)) {
            ScaledComplexProduct(aRe, aIm, bRe, bIm, productRe, productIm);
        } else {
            const float sRe = alphaRe * aRe - alphaIm * aIm;
            const float sIm = alphaRe * aIm + alphaIm * aRe;
            productRe = sRe * bRe - sIm * bIm;
            productIm = sRe * bIm + sIm * bRe;
        }
        accRe += productRe;
        accIm += productIm;
    }

    __aicore__ inline void AddGenericBeta(uint64_t out, float &accRe, float &accIm)
    {
        if (skipBeta) {
            return;
        }
        const uint64_t oi = out * 2;
        const float cRe = cGm.GetValue(oi);
        const float cIm = cGm.GetValue(oi + 1);
        accRe += betaRe * cRe - betaIm * cIm;
        accIm += betaRe * cIm + betaIm * cRe;
    }

    __aicore__ inline void InitGenericLocalBuffers()
    {
        pipe.InitBuffer(genericOutBuf, 8 * sizeof(float));
    }

    __aicore__ inline void StoreGeneric(uint64_t out, float re, float im)
    {
        LocalTensor<float> local = genericOutBuf.Get<float>();
        local.SetValue(0, re);
        local.SetValue(1, im);
        auto ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ready);
        WaitFlag<HardEvent::S_MTE3>(ready);
        DataCopyParams copyParams = {1, static_cast<uint16_t>(2 * sizeof(float)), 0, 0};
        DataCopyPad(yGm[out * 2], local, copyParams);
        auto done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(done);
        WaitFlag<HardEvent::MTE3_S>(done);
    }

    __aicore__ inline void AccumulateGenericRow(
        int32_t row, uint32_t col, float &accRe, float &accIm)
    {
        const uint32_t rs =
            static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseOffset);
        const uint32_t re =
            static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseOffset);
        for (uint32_t idx = rs; idx < re; idx++) {
            const uint32_t k = static_cast<uint32_t>(
                colIndicesGm.GetValue(idx) - idxBaseOffset);
            AccumulateGenericProduct(idx, k, col, accRe, accIm);
        }
    }

    __aicore__ inline void AccumulateGenericColumn(
        int32_t row, uint32_t col, float &accRe, float &accIm)
    {
        const uint32_t begin =
            static_cast<uint32_t>(transposeRowOffsetsGm.GetValue(row));
        const uint32_t end =
            static_cast<uint32_t>(transposeRowOffsetsGm.GetValue(row + 1));
        for (uint32_t position = begin; position < end; position++) {
            const uint32_t k =
                static_cast<uint32_t>(transposeSourceRowsGm.GetValue(position));
            const uint32_t index =
                static_cast<uint32_t>(transposePermutationGm.GetValue(position));
            AccumulateGenericProduct(index, k, col, accRe, accIm);
        }
    }

    __aicore__ inline void ProcessGeneric()
    {
        for (uint32_t ri = myRowStart; ri < myRowEnd; ri++) {
            const int32_t row = reorderGm.GetValue(ri);
            for (uint32_t col = 0; col < N; col++) {
                float accRe = 0.0f;
                float accIm = 0.0f;
                if (opA == 0) {
                    AccumulateGenericRow(row, col, accRe, accIm);
                } else {
                    AccumulateGenericColumn(row, col, accRe, accIm);
                }
                const uint64_t out = GenericCIndex(static_cast<uint32_t>(row), col);
                AddGenericBeta(out, accRe, accIm);
                StoreGeneric(out, accRe, accIm);
            }
        }
    }

    __aicore__ inline void ProcessChunk(uint32_t colStart, uint32_t chunkLen, uint32_t chunkAlign)
    {
        if (rowGroup > 0) {
            ProcessChunkGrouped(colStart, chunkLen, chunkAlign);
            return;
        }
        DataCopyParams copyParams = {1, (uint16_t)(chunkLen * 2 * sizeof(float)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};

        // 交织域宽度：一个 complex 元素占 2 个 float。
        uint32_t aosLen = chunkAlign * 2;

        LocalTensor<float> accA = accABuf.Get<float>();
        LocalTensor<float> accB = accBBuf.Get<float>();
        LocalTensor<float> swapped = swappedBuf.Get<float>();
        LocalTensor<float> sign = signBuf.Get<float>();
        LocalTensor<uint32_t> pairOffset = pairOffsetBuf.Get<uint32_t>();

        for (uint32_t ri = myRowStart; ri < myRowEnd; ri++) {
            int32_t row = reorderGm.GetValue(ri);
            uint32_t rs = static_cast<uint32_t>(rowOffsetsGm.GetValue(row) - idxBaseOffset);
            uint32_t re = static_cast<uint32_t>(rowOffsetsGm.GetValue(row + 1) - idxBaseOffset);
            uint64_t outBase = (static_cast<uint64_t>(row) * N + colStart) * 2;

            // 上一行末尾读 accA/accB 的是 Gather/Add(PIPE_V)，这一行开头写
            // accA/accB 的是 Duplicate(PIPE_V)，同一条流水线上的顺序指令，
            // 不需要额外跨引擎同步。
            Duplicate(accA, 0.0f, aosLen);
            Duplicate(accB, 0.0f, aosLen);
            fastAlphaRangeLost = false;

            if (rs < re) {
                if (batchSlots > 0) {
                    AccumulateRowBatched(accA, accB, rs, re, colStart, aosLen,
                        copyParams, padParams);
                } else {
                    AccumulateRow(accA, accB, rs, re, colStart, aosLen, copyParams, padParams);
                }
            }
            if (!skipBeta) {
                AccumulateBeta(accA, accB, outBase, aosLen, copyParams, padParams);
            }
            StoreRow(accA, accB, swapped, sign, pairOffset, outBase, aosLen, copyParams,
                row, colStart, chunkLen);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, SPMM_C64_BUFFER_NUM> rawQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> accABuf, accBBuf;
    TBuf<TPosition::VECCALC> swappedBuf, signBuf;
    TBuf<TPosition::VECCALC> pairOffsetBuf;
    TBuf<TPosition::VECCALC> genericOutBuf;
    GlobalTensor<int32_t> transposeRowOffsetsGm;
    GlobalTensor<int32_t> transposeSourceRowsGm;
    GlobalTensor<int32_t> transposePermutationGm;
    TBuf<TPosition::VECCALC> bBatchBuf;   // B 行槽位，见 AccumulateRowBatched
    TBuf<TPosition::VECCALC> accAGroupBuf, accBGroupBuf;  // 见 ProcessChunkGrouped

    GlobalTensor<int32_t> colIndicesGm;
    GlobalTensor<float> valuesGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    GlobalTensor<float> yGm;

    float alphaRe, alphaIm, betaRe, betaIm;
    uint32_t aRows;
    int32_t ldb, ldc, bOrder, cOrder, opA, opB;
    bool genericPath;
    bool numericGuard;
    bool fastAlphaRangeLost;  // 快路径提前合并 alpha 时溢出/下溢，行末回退 generic
};

__global__ __aicore__ void spmm_arch22_complex64(
    GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tilingGm)
{
    // 见 spmm_kernel.h 中 DEFINE_SPMM_KERNEL 上方对 AIV_ONLY 的说明：
    // 本 kernel 同样只用 Vector 引擎，标记后可按 Vector Core 数起 block。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmmArch22KernelComplex64 op;
    op.Init(rowOff, colIdx, values, matB, matC, workspace, tilingGm);
    op.Process();
}

__global__ __aicore__ void spmm_arch22_complex64_out(
    GM_ADDR rowOff, GM_ADDR colIdx, GM_ADDR values,
    GM_ADDR matB, GM_ADDR matC, GM_ADDR matY, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmmArch22KernelComplex64 op;
    op.Init(rowOff, colIdx, values, matB, matC, matY, workspace, tilingGm);
    op.Process();
}
