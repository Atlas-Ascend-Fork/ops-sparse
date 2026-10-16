/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the "License".
 * Please refer to the License for the details.
 * ----------------------------------------------------------------------------------------------------------
 */

/*
 * gather Kernel — Atlas A2/A3 Vector Core (AIV)
 *
 * 性能关键：Y 为稠密向量，X.values 为稀疏向量（长度 nnz）。公式 X.values[i] = Y[indices[i]-idxBase]。
 * indices 任意（随机）分布，因此逐元素从 Y GM 随机读会被 MTE 按 32 字节突发补齐，产生带宽放大与
 * 极大的 per-DataCopy 开销。本实现改为：
 *   1) indices(GM) --MTE2--> idxUB（连续读，一次搬运）
 *   2) 将 Y 按 UB 容量分块（分块容量由 UB 总预算扣除各队列后反推），每块 Y[cstart:cend) --MTE2-->
 *      yUB（连续读，合并成极少次大块搬运；整块 32B 对齐、尾块精确长度，既不越界也无对齐歧义）
 *   3) 对 tile 内每个输出元素 j，若其源位置落在当前 Y 块内，则把 yUB 中对应字节经标量 SetValue 搬入
 *      输出槽位 j*elemBytes（UB->UB 标量访问无 32 字节对齐要求，bit-wise exact）
 *   4) X.values --MTE3--> GM（core 内输出连续，每 tile 一次）
 *
 * 纯数据搬运（无算术），逐元素 bit-wise exact；complex64 按 8 字节整体搬。
 */

#include "kernel_operator.h"
#include "gather_tiling_data.h"
#include "gather_kernel.h"

using namespace AscendC;

#define BUFFER_NUM 2

// 32 字节向上对齐（UB 侧搬运的对齐单位；DataCopyPad 的 UB 块跨度须按 32B 对齐）。
#define GATHER_ALIGN32(x) (((x) + 31U) & ~static_cast<uint32_t>(31U))

class GatherKernel {
public:
    __aicore__ inline GatherKernel()
    {}

    __aicore__ inline void Init(GM_ADDR idxGmAddr, GM_ADDR yGmAddr, GM_ADDR xGmAddr,
                                const GatherTilingData &tiling)
    {
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        if (blockIdx >= tiling.blockNum) {
            return;
        }
        coreNnz_    = tiling.coreNnzCount[blockIdx];
        coreOffset_ = tiling.coreNnzOffset[blockIdx];
        elemBytes_  = tiling.elemBytes;
        yElems_     = tiling.yElems;
        idxBase_    = tiling.idxBase;

        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxGmAddr), tiling.nnz);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(yGmAddr), tiling.yElems * tiling.elemBytes);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(xGmAddr), tiling.nnz * tiling.elemBytes);

        if (coreNnz_ == 0) {
            return;
        }

        uint32_t tileNn = tiling.tileNn;
        // UB 容量核算（PR 检视 [重要]）：所有 InitBuffer 的合计容量必须 <= GATHER_UB_BUDGET_BYTES。
        //   idx 队列 = BUFFER_NUM * tileNn * sizeof(int32_t)；val 队列 = align32(tileNn * elemBytes)；
        //   Y 缓冲   = 预算 - 上面两项 —— 即「缩小 Y chunk」而非超配，保证 complex64 +
        //   tileNn = GATHER_TILE_NN_MAX 的最坏组合也不会撑爆 UB（A3 实测可用 UB ≈ 191.75 KiB）。
        uint32_t idxBytes = tileNn * sizeof(int32_t);
        uint32_t outBytes = GATHER_ALIGN32(tileNn * elemBytes_);
        uint32_t queueBytes = idxBytes * BUFFER_NUM + outBytes;
        uint32_t yBytesBudget =
            (queueBytes >= GATHER_UB_BUDGET_BYTES) ? 0U : (GATHER_UB_BUDGET_BYTES - queueBytes);

        // Y 分块容量（每趟 pass 读入 UB 的元素数）：受剩余 UB 预算约束，向下取整到 16 的倍数
        // （elemBytes 为 2/4/8，故 capElems * elemBytes 恒为 32 字节倍数）。容量尽量大 => pass 数少。
        uint32_t capElems = (yBytesBudget / elemBytes_) & ~static_cast<uint32_t>(15);
        if (capElems > yElems_) {
            capElems = yElems_;
        }
        if (capElems == 0U) {
            // 防御性下限：仅在 tileNn/elemBytes 组合异常（队列已占满预算）时触发，
            // 保证 Process 的 Y 分块循环必然推进。
            capElems = 1U;
        }
        ubCapElems_ = capElems;

        uint32_t yBytes = GATHER_ALIGN32(capElems * elemBytes_);   // 32 字节对齐

        pipe.InitBuffer(inQueueIdx, BUFFER_NUM, idxBytes);
        pipe.InitBuffer(yQueue, 1, yBytes);
        pipe.InitBuffer(valQueue, 1, outBytes);

        tileNn_ = tileNn;
    }

    __aicore__ inline void Process()
    {
        if (coreNnz_ == 0) {
            return;
        }
        uint32_t tileNum = (coreNnz_ + tileNn_ - 1) / tileNn_;
        for (uint32_t tile = 0; tile < tileNum; tile++) {
            Compute(tile);
        }
    }

private:
    // 块内 scatter：把索引以 int64 成对读取（一次 8 字节加载覆盖 2 个 int32 索引），
    // 把内层最热的「逐元素读索引」标量访存次数减半（原每元素 1 次 int32 读 → 现每 2 元素 1 次 int64 读）；
    // 减 base / 两次范围 clamp 为寄存器运算（廉价），按元素原生类型（int16/int32/int64）整元素搬移，
    // bit-wise exact；UB 标量访问无 32B 对齐要求，故无需对源/目的地址做对齐处理。
    template <typename T>
    __aicore__ inline void ScatterPass(LocalTensor<uint8_t> yLocal, LocalTensor<uint8_t> outBuf,
                                       LocalTensor<int64_t> idxI64, uint32_t curTileNn,
                                       uint64_t cstart, uint64_t cend)
    {
        auto yL = yLocal.ReinterpretCast<T>();
        auto oB = outBuf.ReinterpretCast<T>();
        uint32_t j = 0;
        for (; j + 1 < curTileNn; j += 2) {
            int64_t pair = idxI64.GetValue(j >> 1);
            int32_t sa = static_cast<int32_t>(pair & 0xFFFFFFFFULL);
            int32_t sb = static_cast<int32_t>(pair >> 32);
            sa -= static_cast<int32_t>(idxBase_);
            if (sa < 0) {
                sa = 0;
            }
            if (static_cast<uint32_t>(sa) >= yElems_) {
                sa = static_cast<int32_t>(yElems_) - 1;
            }
            uint64_t posA = static_cast<uint32_t>(sa);
            if (posA >= cstart && posA < cend) {
                oB.SetValue(j, yL.GetValue(static_cast<uint32_t>(posA - cstart)));
            }
            sb -= static_cast<int32_t>(idxBase_);
            if (sb < 0) {
                sb = 0;
            }
            if (static_cast<uint32_t>(sb) >= yElems_) {
                sb = static_cast<int32_t>(yElems_) - 1;
            }
            uint64_t posB = static_cast<uint32_t>(sb);
            if (posB >= cstart && posB < cend) {
                oB.SetValue(j + 1, yL.GetValue(static_cast<uint32_t>(posB - cstart)));
            }
        }
        if (j < curTileNn) {
            // 奇数元素收尾：切回 int32 视图读取
            int32_t s = idxI64.ReinterpretCast<int32_t>().GetValue(j) - static_cast<int32_t>(idxBase_);
            if (s < 0) {
                s = 0;
            }
            if (static_cast<uint32_t>(s) >= yElems_) {
                s = static_cast<int32_t>(yElems_) - 1;
            }
            uint64_t pos = static_cast<uint32_t>(s);
            if (pos >= cstart && pos < cend) {
                oB.SetValue(j, yL.GetValue(static_cast<uint32_t>(pos - cstart)));
            }
        }
    }

    // 1) 读本 tile 的 indices（连续，一次 MTE2）
    __aicore__ inline LocalTensor<int32_t> LoadIdx(uint32_t localOffset, uint32_t curTileNn)
    {
        auto idxLocal = inQueueIdx.AllocTensor<int32_t>();
        DataCopyPad(idxLocal, idxGm[coreOffset_ + localOffset],
            {1, static_cast<uint16_t>(curTileNn * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        return idxLocal;
    }

    // 2) 单趟 Y 分块搬运：将 Y[cstart, cend) 连续读入 yLocal。
    // 单趟 Y 拆成若干连续块；用 blockCount>1 的 DataCopyPad 单次启动搬运，避免逐块启动的高昂
    // per-DataCopy 开销。两条硬约束（PR 检视 [重要]，以及前述 GM 越界读意见）：
    //   a) GM 侧读取长度必须精确（传实际字节数），绝不越界读 Y 分配范围之外 → 不对读取长度补齐；
    //   b) UB 侧 block 跨度须 32 字节对齐。若 blockLen 非 32B 对齐，「紧凑布局」与「每块补齐到
    //      32B」两种语义不一致，多块场景下会跨块错读（如 blockLen=58027 时尾块起于 116054，
    //      模 32 余 22）。故整块 blockLen 取 32B 倍数（且 <= GATHER_DC_BLOCK_BYTES_MAX 的 uint16
    //      上限），此时两种语义完全等价；尾块起始偏移 = fullBlocks * blockLen 亦天然 32B 对齐。
    // 读满 (fullBlocks) 个对齐整块，再单独发一次精确长度的尾块（blockCount=1，长度不补齐，
    // 故绝不越界；UB 侧仅缓冲容量按 32B 对齐，见 Init 中的 yBytes）。
    __aicore__ inline void LoadYChunk(LocalTensor<uint8_t> yLocal, uint64_t cstart, uint64_t cend)
    {
        uint64_t passElems = cend - cstart;
        uint32_t passBytes = static_cast<uint32_t>(passElems * elemBytes_);
        if (passBytes == 0U) {
            return;
        }
        uint32_t nBlocks = (passBytes + GATHER_DC_BLOCK_BYTES_MAX - 1U) / GATHER_DC_BLOCK_BYTES_MAX;
        uint32_t blockLen = GATHER_ALIGN32((passBytes + nBlocks - 1U) / nBlocks);
        nBlocks = (passBytes + blockLen - 1U) / blockLen;
        uint32_t fullBlocks = nBlocks - 1U;
        if (fullBlocks > 0U) {
            DataCopyPad(yLocal, yGm[static_cast<uint64_t>(cstart) * elemBytes_],
                {static_cast<uint16_t>(fullBlocks), static_cast<uint16_t>(blockLen), 0, 0},
                {false, 0, 0, 0});
        }
        uint32_t remBytes = passBytes - fullBlocks * blockLen;
        if (remBytes > 0U) {
            DataCopyPad(yLocal[fullBlocks * blockLen],
                yGm[static_cast<uint64_t>(cstart) * elemBytes_ + fullBlocks * blockLen],
                {1, static_cast<uint16_t>(remBytes), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE2>();
    }

    // 3) 块内 scatter：按元素字宽选择 ScatterPass 的原生标量视图
    __aicore__ inline void ScatterDispatch(LocalTensor<uint8_t> yLocal, LocalTensor<uint8_t> outBuf,
                                           LocalTensor<int64_t> idxI64, uint32_t curTileNn,
                                           uint64_t cstart, uint64_t cend)
    {
        if (elemBytes_ == 2) {
            ScatterPass<int16_t>(yLocal, outBuf, idxI64, curTileNn, cstart, cend);
        } else if (elemBytes_ == 4) {
            ScatterPass<int32_t>(yLocal, outBuf, idxI64, curTileNn, cstart, cend);
        } else {
            ScatterPass<int64_t>(yLocal, outBuf, idxI64, curTileNn, cstart, cend);
        }
    }

    __aicore__ inline void Compute(uint32_t tile)
    {
        uint32_t localOffset = tile * tileNn_;
        uint32_t curTileNn   = coreNnz_ - localOffset;
        if (curTileNn > tileNn_) {
            curTileNn = tileNn_;
        }

        // 1) 读本 tile 的 indices（连续，一次 MTE2）
        auto idxLocal = LoadIdx(localOffset, curTileNn);
        auto outBuf = valQueue.AllocTensor<uint8_t>();
        uint32_t outBytes = curTileNn * elemBytes_;

        // 2)+3) 分块读 Y 并在块内 scatter（idxLocal 在 ScatterPass 内减 base + clamp）
        for (uint64_t cstart = 0; cstart < yElems_; cstart += ubCapElems_) {
            uint64_t cend = cstart + ubCapElems_;
            if (cend > yElems_) {
                cend = yElems_;
            }
            auto yLocal = yQueue.AllocTensor<uint8_t>();
            LoadYChunk(yLocal, cstart, cend);
            auto idxI64 = idxLocal.ReinterpretCast<int64_t>();
            ScatterDispatch(yLocal, outBuf, idxI64, curTileNn, cstart, cend);
            yQueue.FreeTensor(yLocal);
        }

        // 4) 连续写回 X.values（core 内输出连续），每 tile 一次 MTE3
        PipeBarrier<PIPE_V>();
        uint64_t xByte = static_cast<uint64_t>(coreOffset_ + localOffset) * elemBytes_;
        DataCopyPad(xGm[xByte], outBuf,
            {1, static_cast<uint16_t>(outBytes), 0, 0, 0});
        PipeBarrier<PIPE_MTE3>();

        valQueue.FreeTensor(outBuf);
        inQueueIdx.FreeTensor(idxLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueIdx;
    TQue<TPosition::VECOUT, 1> yQueue;
    TQue<TPosition::VECOUT, 1> valQueue;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<uint8_t> yGm;
    GlobalTensor<uint8_t> xGm;
    uint32_t coreNnz_;
    uint32_t coreOffset_;
    uint32_t elemBytes_;
    uint64_t yElems_;
    uint32_t idxBase_;
    uint32_t tileNn_;
    uint32_t ubCapElems_;
};

__global__ __vector__ void gather_custom(
    GM_ADDR idxGm, GM_ADDR yGm, GM_ADDR xGm, const GatherTilingData tiling)
{
    GatherKernel op;
    op.Init(idxGm, yGm, xGm, tiling);
    op.Process();
}

void gather_kernel_do(
    GM_ADDR idxDev, GM_ADDR yDev, GM_ADDR xValuesDev,
    const GatherTilingData &tiling, uint32_t blockNum, void *stream)
{
    gather_custom<<<blockNum, nullptr, stream>>>(
        idxDev, yDev, xValuesDev, tiling);
}
