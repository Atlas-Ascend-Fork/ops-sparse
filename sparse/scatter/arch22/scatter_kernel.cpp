/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * scatter Kernel — Template D: no in-kernel sort + DoubleBuffer
 *
 * 能力扩展：单模板族 KernelScatter<T, ELEM_BYTES>，
 * scatter_kernel_do 按 aclDataType 分发 5 个实例化：
 *   int8      -> KernelScatter<int8_t,     1>
 *   fp16      -> KernelScatter<half,       2>
 *   bf16      -> KernelScatter<bfloat16_t, 2>
 *   fp32      -> KernelScatter<float,      4>
 *   complex64 -> KernelScatter<float,      8>   (2×fp32 交错存储，纯字节搬运不拆实虚部)
 *
 * 统一口径：
 *   - valGm / yGm 以 T（如 complex64 时 T=float）视角绑定与偏移，
 *     GM 偏移乘 GM_STRIDE = ELEM_BYTES / sizeof(T)（complex64 为 2，其余为 1）；
 *   - yGm 绑定长度 = yLen / sizeof(T)（complex64 时 = 2×size）；
 *   - UB 内 run 检测按逻辑元素（= T 元素 / GM_STRIDE）进行，WriteRun 内部
 *     统一换算为 T 元素偏移/长度后再 DataCopyPad（段路径）/ SetValue（单元素路径）。
 *
 * 写路径约束：所有 GM 写一律 UB 暂存 + MTE3 DataCopyPad（含段长 1 单元素写），
 * 禁止 GlobalTensor::SetValue（对 MTE2 不可见）。
 *
 * 段路径：整段走纯 MTE 双跳 DataCopyPad（GM→VECOUT 直读 + VECOUT→GM 写出，
 * 均为 A2/A3 文档通路），不使用 V pipe 指令级 Copy<T>：Copy 指令在
 * dav-2201(910B3) 上存在 arch 特异静默缺陷（int8 mask 语义错误；V→MTE3
 * 非确定可见性竞态），大连续 case fp32 会 bit-exact 丢值。
 */

#include "kernel_operator.h"
#include "scatter.h"
#include "scatter_kernel.h"

#include <cstdio>

using namespace AscendC;

template <typename T, uint32_t ELEM_BYTES>
class KernelScatter {
public:
    __aicore__ inline KernelScatter()
    {}

    __aicore__ inline void Init(GM_ADDR valGmAddr, GM_ADDR idxGmAddr,
                                GM_ADDR yGmAddr, const ScatterTilingData &tiling)
    {
        // 多余 block 防护：先清零本核计数再判越界，避免早退路径读到未初始化成员
        coreNnz_ = 0;
        coreOffset_ = 0;
        tileNn_ = 0;
        idxBase_ = 0;
        slot_ = 0;
        slotPending_ = 0;

        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        if (blockIdx >= tiling.blockNum) {
            return;
        }
        coreNnz_    = tiling.coreNnzCount[blockIdx];
        coreOffset_ = tiling.coreNnzOffset[blockIdx];
        idxBase_    = tiling.idxBase;

        // T 视角绑定（complex64: T=float，长度 = 字节数/sizeof(float)）；
        // nnz×ELEM_BYTES 按 uint64 计算：uint32 乘法在 complex64(nnz≥2^29)/
        // fp32(nnz≥2^30) 时回绕，host 上界 INT32_MAX 拦不住
        valGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(valGmAddr),
                              (static_cast<uint64_t>(tiling.nnz) * ELEM_BYTES) / sizeof(T));
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxGmAddr), tiling.nnz);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yGmAddr), tiling.yLen / sizeof(T));

        if (coreNnz_ == 0) {
            return;
        }

        tileNn_ = tiling.tileNn;

        // 索引缓冲与 dtype 无关恒 4B/元素；values/out 缓冲按 ELEM_BYTES
        pipe.InitBuffer(inQueueVal, SCATTER_BUFFER_NUM, tiling.tileNnAligned * ELEM_BYTES);
        pipe.InitBuffer(inQueueIdx, SCATTER_BUFFER_NUM, tiling.tileNnAligned * sizeof(int32_t));
        pipe.InitBuffer(outQueue, 1, tiling.tileNnAligned * ELEM_BYTES);
        // 单元素写 K 槽 ring 暂存（K×32B，仅作 MTE3 源；每槽 32B 对齐满足 MTE3
        // 源地址对齐要求，槽内按 ELEM_BYTES 最大 8B 切槽足够容纳任一 dtype 单元素）
        pipe.InitBuffer(scratchBuf_, SCRATCH_SLOTS * SLOT_BYTES);
        scratch_ = scratchBuf_.Get<T>(SCRATCH_SLOTS * SLOT_BYTES / sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNum = (coreNnz_ + tileNn_ - 1) / tileNn_;

        {
            uint32_t curTileNn = (coreNnz_ < tileNn_) ? coreNnz_ : tileNn_;
            CopyInAsync(coreOffset_, curTileNn);
        }

        for (uint32_t tile = 0; tile < tileNum; tile++) {
            uint32_t localOffset = tile * tileNn_;
            if (localOffset >= coreNnz_) {
                break;
            }
            uint32_t curTileNn = coreNnz_ - localOffset;
            if (curTileNn > tileNn_) {
                curTileNn = tileNn_;
            }

            WaitCopyIn();

            // 双缓冲预取：MTE2 载入下一 tile 与当前 tile 的 V 计算/MTE3 写出重叠
            if (tile + 1 < tileNum) {
                uint32_t nextOffset = coreOffset_ + (tile + 1) * tileNn_;
                uint32_t nextNn     = coreNnz_ - (tile + 1) * tileNn_;
                if (nextNn > tileNn_) {
                    nextNn = tileNn_;
                }
                CopyInAsync(nextOffset, nextNn);
            }

            // tileGmBase：本 tile 首元素的全局逻辑元素序号（段路径 GM 直读源基址）
            Compute(curTileNn, coreOffset_ + localOffset);
        }
    }

private:
    __aicore__ inline void CopyInAsync(uint32_t gmOffset, uint32_t curTileNn)
    {
        // GM 偏移按 T 视角（complex64 ×2）；尾块 blockLen 按实际字节数（可非 32B 对齐）
        auto valLocal = inQueueVal.AllocTensor<T>();
        DataCopyPad(valLocal, valGm[gmOffset * GM_STRIDE],
            {1, static_cast<uint16_t>(curTileNn * ELEM_BYTES), 0, 0},
            {false, 0, 0, 0});
        inQueueVal.EnQue(valLocal);

        auto idxLocal = inQueueIdx.AllocTensor<int32_t>();
        DataCopyPad(idxLocal, idxGm[gmOffset],
            {1, static_cast<uint16_t>(curTileNn * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        inQueueIdx.EnQue(idxLocal);
    }

    __aicore__ inline void WaitCopyIn()
    {
        valBuf_   = inQueueVal.DeQue<T>();
        idxBuf32_ = inQueueIdx.DeQue<int32_t>();
    }

    // run 检测：在换算 idxBase 后的目标索引上做连续段聚簇判断
    __aicore__ inline void Compute(uint32_t curTileNn, uint32_t tileGmBase)
    {
        uint32_t runStart = 0;
        int32_t prevTarget = idxBuf32_.GetValue(0) - idxBase_;

        for (uint32_t i = 1; i <= curTileNn; i++) {
            bool isEnd = (i == curTileNn);
            bool isDisconnected = false;
            int32_t curTarget = 0;

            if (!isEnd) {
                curTarget = idxBuf32_.GetValue(i) - idxBase_;
                isDisconnected = (curTarget != prevTarget + 1);
            }

            if (isEnd || isDisconnected) {
                uint32_t runLen = i - runStart;
                int32_t targetIdx = idxBuf32_.GetValue(runStart) - idxBase_;

                WriteRun(targetIdx, runStart, runLen, tileGmBase);

                runStart = i;
                if (!isEnd) {
                    prevTarget = curTarget;
                }
            } else {
                prevTarget = curTarget;
            }
        }

        // tile 末尾 PIPE_ALL：本 tile 尚在途的 MTE3（scratch 槽单元素写 + 段路径
        // outLocal 写）全部完成后，才把 valBuf_/idxBuf32_ 归还队列（MTE2 复写前
        // MTE3 已读完——双缓冲正确性）；同时 ring 的 pending 清零（该 barrier 已
        // 保证所有槽的在途 MTE3 完成，下一 tile 从槽 0 起安全复用）。
        PipeBarrier<PIPE_ALL>();
        slotPending_ = 0;
        inQueueVal.FreeTensor(valBuf_);
        inQueueIdx.FreeTensor(idxBuf32_);
    }

    // GM 写一律 UB 暂存 + MTE3 DataCopyPad（含段长 1 单元素写），
    // 禁止 GlobalTensor::SetValue 直接写 GM（MTE2 不可见）。
    __aicore__ inline void WriteRun(int32_t targetIdx, uint32_t srcOffset, uint32_t runLen,
                                     uint32_t tileGmBase)
    {
        // 统一换算为 T 元素口径（complex64: 逻辑元素 = 2×float）
        const uint32_t tRunLen = runLen * GM_STRIDE;
        const uint32_t tSrcOff = srcOffset * GM_STRIDE;

        if (runLen == 1) {
            // 单元素路径（乱序索引 runLen==1 退化，1B/2B/4B/8B MTE3 写）：
            // K 槽 ring 暂存摊销固定开销。值写 32B 对齐 scratch 槽（complex64 经
            // GM_STRIDE 写实虚两个 float），每元素仅一次 MTE3 DataCopyPad（乱序
            // 散布写无法合并 MTE3，硬件边界）；slot 回绕复用前（发满 K 个 MTE3）
            // 才 PipeBarrier<PIPE_ALL> 一次，保证 MTE3 读完该槽后 V 侧才复写，
            // barrier 按批摊销。
            if (slotPending_ == SCRATCH_SLOTS) {
                PipeBarrier<PIPE_ALL>();
                slotPending_ = 0;
            }
            const uint32_t slotOff = slot_ * (SLOT_BYTES / sizeof(T));
            for (uint32_t j = 0; j < GM_STRIDE; j++) {
                scratch_.SetValue(slotOff + j, valBuf_.GetValue(tSrcOff + j));
            }
            // 偏移按 uint64 换算后再下标：GM_STRIDE=2（complex64）时
            // uint32 乘法在 targetIdx/tileGmBase≥2^30 回绕
            DataCopyPad(yGm[static_cast<uint64_t>(static_cast<uint32_t>(targetIdx)) * GM_STRIDE],
                scratch_[slotOff], {1, static_cast<uint32_t>(ELEM_BYTES), 0, 0, 0});
            slot_ = (slot_ + 1) % SCRATCH_SLOTS;
            slotPending_++;
            return;
        }

        auto outLocal = outQueue.AllocTensor<T>();

        {
            // 段路径（纯 MTE 双跳）：整段 GM(valGm)→VECOUT(outLocal) DataCopyPad
            // 直读 + VECOUT→GM(yGm) 写出，两跳均为 A2/A3 文档通路（DataCopyPad
            // 通路表：GM→VECOUT / VECOUT→GM），不使用 V pipe 指令级 Copy<T>：
            //   1) Copy 指令在 dav-2201(910B3) 上存在 arch 特异静默缺陷（int8
            //      mask 语义错误；V→MTE3 非确定可见性竞态），大连续 case fp32
            //      会 bit-exact 丢值；
            //   2) DataCopyPad blockLen 按字节、无对齐/整块约束，对齐/非对齐 run
            //      统一走此路径，尾部无需逐元素兜底；int8 无 Copy mask 语义
            //      问题（纯字节粒度搬运，五 dtype 同构）。
            // GM 直读源 = 本 tile 全局逻辑基址（tileGmBase）换算 T 元素偏移；
            // 对 valGm 的二次读取通常被 L2 吸收（本 tile 刚由 MTE2 整块读过，
            // tile ≤ tileNnAligned×ELEM_BYTES ≤ 32KB 量级）。
            DataCopyPad(outLocal, valGm[static_cast<uint64_t>(tileGmBase) * GM_STRIDE + tSrcOff],
                {1, static_cast<uint16_t>(tRunLen * sizeof(T)), 0, 0},
                {false, 0, 0, 0});
            // PIPE_ALL：MTE2（GM→VECOUT 写 outLocal）对 MTE3 写出可见（跨 pipe 顺序）
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(yGm[static_cast<uint64_t>(static_cast<uint32_t>(targetIdx)) * GM_STRIDE], outLocal,
                {1, static_cast<uint32_t>(tRunLen * sizeof(T)), 0, 0, 0});
        } // 段路径块结束（outLocal 仅段路径使用）

        // PIPE_ALL：MTE3 写出完成后再回收缓冲
        PipeBarrier<PIPE_ALL>();
        outQueue.FreeTensor(outLocal);
    }

private:
    static constexpr uint32_t GM_STRIDE = ELEM_BYTES / sizeof(T);   // complex64: 2, 其余: 1
    // 单元素写 ring 暂存参数：K 槽 × 每槽 32B（MTE3 源 32B 对齐要求；槽容量 32B
    // 足够任一 dtype 单元素 ≤8B）。K 取 64：batch 内 barrier 次数 = 元素数/64，
    // UB 占用仅 2KB（tileNn 最大 4K×8B=32KB 量级，可忽略）。
    static constexpr uint32_t SCRATCH_SLOTS = 64;
    static constexpr uint32_t SLOT_BYTES = 32;

    TPipe pipe;
    TQue<TPosition::VECIN, SCATTER_BUFFER_NUM> inQueueVal;
    TQue<TPosition::VECIN, SCATTER_BUFFER_NUM> inQueueIdx;
    TQue<TPosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> scratchBuf_;   // 单元素写 K 槽 ring 暂存（仅 MTE3 源）
    GlobalTensor<T> valGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<T> yGm;
    LocalTensor<T> valBuf_;
    LocalTensor<int32_t> idxBuf32_;
    LocalTensor<T> scratch_;
    uint32_t slot_ = 0;          // ring 当前槽下标
    uint32_t slotPending_ = 0;   // 已发出 MTE3 但未 barrier 的槽数
    uint32_t coreNnz_;
    uint32_t coreOffset_;
    uint32_t tileNn_;
    int32_t idxBase_;
};

template <typename T, uint32_t ELEM_BYTES>
__global__ __vector__ void scatter_custom(
    GM_ADDR valGm, GM_ADDR idxGm, GM_ADDR yGm, const ScatterTilingData tiling)
{
    KernelScatter<T, ELEM_BYTES> op;
    op.Init(valGm, idxGm, yGm, tiling);
    op.Process();
}

void scatter_kernel_do(
    void *valDev, void *idxDev, void *yDev, aclDataType valueType,
    const ScatterTilingData &tiling, uint32_t blockNum, void *stream)
{
    // Host 校验链已保证 valueType ∈ 五 dtype 集合，default 不达
    switch (valueType) {
        case ACL_INT8:
            scatter_custom<int8_t, 1><<<blockNum, nullptr, stream>>>(
                (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
            break;
        case ACL_FLOAT16:
            scatter_custom<half, 2><<<blockNum, nullptr, stream>>>(
                (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
            break;
        case ACL_BF16:
            scatter_custom<bfloat16_t, 2><<<blockNum, nullptr, stream>>>(
                (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
            break;
        case ACL_FLOAT:
            scatter_custom<float, 4><<<blockNum, nullptr, stream>>>(
                (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
            break;
        case ACL_COMPLEX64:
            // complex64 = 2×fp32 交错存储：T=float、ELEM_BYTES=8，纯字节搬运不拆实虚部
            scatter_custom<float, 8><<<blockNum, nullptr, stream>>>(
                (GM_ADDR)valDev, (GM_ADDR)idxDev, (GM_ADDR)yDev, tiling);
            break;
        default:
            // Host 校验链保证 valueType ∈ 五 dtype 集合，default 不达；
            // 防御深度：未来 dtype 扩展遗漏校验时显式暴露，禁止静默不发射 kernel
            // （成功但无效果的假成功）。
            fprintf(stderr,
                    "aclsparseScatter: scatter_kernel_do unsupported valueType %d, kernel not launched\n",
                    static_cast<int>(valueType));
            break;
    }
}
