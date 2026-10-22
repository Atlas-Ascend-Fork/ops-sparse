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

#ifndef SPMM_ARCH22_H
#define SPMM_ARCH22_H

#include <stdint.h>

#ifndef __gm__
#define __gm__
#endif

/* host 与 device 共用的内联工具函数限定符，用于下面那几个 UB 预算函数——它们必须
 * 被 spmm_host.cpp 和 kernels/spmm_kernel.h 调用同一份实现，否则两侧的 chunk 边界
 * 会漂移。
 *
 * cce 是单源模型：host 与 device 代码在同一个编译单元里，靠属性而不是预处理宏区分，
 * 所以 #ifdef __CCE_AICORE__ 之类的判别式在这里无效（编译 spmm_host.cpp 时它同样
 * 有定义）。只标 __aicore__ 会让 host 调用报 "call to __aicore__ function from
 * __host__ function"，不标则让 kernel 调用报 "no matching function for call to"。
 * __host_aicore__（asc/impl/utils/sys_macros.h 里的 [host, aicore]）正是两端都可调
 * 的限定符。#ifdef 兜底是因为本文件也会被不含 AscendC 头的纯 host 代码包含，同
 * 文件顶部 __gm__ 的处理口径一致。 */
#ifdef __host_aicore__
#define SPMM_ARCH22_HD static __host_aicore__ inline
#else
#define SPMM_ARCH22_HD static inline
#endif

#define SPMM_ARCH22_BUFFER_NUM     2
/* bQueue 的最大预取深度，也就是同时在飞的 B 行 DMA 数上限。
 *
 * 深度 8 实测过，全线变慢约 10%（P-02 fp32 5401->6032us、P-01 5.4%、CTL 9%），
 * 所以退回 2。结论是 B 行 gather 不是发起延迟受限：多排队并不能让 MTE2 更快出数，
 * 而 TQue 每个缓冲槽的事件簿记是实打实的额外开销。
 *
 * 需要留意的是这个结论依赖当前 benchmark 的稀疏模式——perf8_bench 沿用官方
 * build_regular_csr 的 (row*131 + j) % K，同一行的列索引是连续整数，B 行在内存里
 * 因此完全连续，本来就是 DMA 最友好的情形，深预取没有延迟可藏。若换成 P 表真正
 * 对应的真实图数据（Cora / ogbn-arxiv / ogbn-products，列索引散乱），这个深度值
 * 需要重新标定。 */
#define SPMM_ARCH22_BQUEUE_MAX     2
#define SPMM_ARCH22_UB_SIZE        (192 * 1024)
/* 12KB 而非 8KB：多出的 4KB 覆盖 kernels/spmm_kernel.h 里 SPMM_ARCH22_NZ_TILE 那
 * 三块定长 tile（列索引 1KB + 非零值 1KB + 窄类型的 fp32 转换暂存 1KB）。它们的
 * 大小与 N 无关，所以计入 overhead 而不是计入 bytesPerElem，host 侧
 * SpmmArch22ComputeNChunks 的每元素口径因此不必改动。 */
#define SPMM_ARCH22_UB_OVERHEAD    (12 * 1024)
#define SPMM_ARCH22_ALIGN          8
/* DataCopyPad 的 UB 侧起始地址必须 32 字节对齐。 */
#define SPMM_ARCH22_UB_ADDR_ALIGN  32

/* 一批整块搬入 UB 的非零元个数（列索引 + 值）。取 256 是因为它远大于稀疏矩阵
 * 常见的行度数，绝大多数行一批即可装完，跨批边界的预取气泡可忽略；同时三块
 * tile 合计只占 3KB，不挤压按 N 伸缩的那部分 UB 预算。 */
#define SPMM_ARCH22_NZ_TILE        256

/* AccumulateRowBatched 一批处理的非零元个数上限，也就是 UB 里 B 行槽位的个数上限。
 * 实际值由 InitLocalBuffers 按 UB 余量在 {32,16,8} 里自适应挑（槽位数组按 N 伸缩），
 * 装不下 8 个就退回逐元素路径。
 *
 * 上限取 32 是因为它已覆盖 P-03 的行度数 25.3（整行一批装完）；实测把它固定成 32
 * 会让 N=1433 的 P-01 因 UB 装不下而整体退回旧路径(55->69us)，所以必须自适应而不是
 * 写死。下限 8 覆盖 P-02 的行度数 6.9，实测 8 与 16 在 P-02 上没有差别。 */
#define SPMM_ARCH22_BATCH_SLOTS_MAX 32
#define SPMM_ARCH22_BATCH_SLOTS_MIN 8

/* 一组一起处理的行数上限，见 kernels/spmm_kernel.h 的 ProcessChunkGrouped。
 *
 * 逐行处理时每行都要各付一遍固定开销：3 次 GM 标量读(reorder + rowOffsets 两次)、
 * 2 次 GM DMA(C 读入、C 写出)、2~3 组 TQue 同步往返。行度数小的时候这笔开销摊不
 * 下去——用探针把每非零元的 B 行 DMA 与两次标量读全部去掉后，P-02 尺寸/nnz 对齐
 * 的规则合成输入中，fp32 只从 7044us 降到 5601us，剩下的 5601us 折算成每行
 * 1323ns；再用 P-02
 * 与 P-03 两点拟合 cost = a*行数 + b*非零元数，得 a=1392ns、b=39ns(N=128)，与探针
 * 独立吻合。也就是说 P-02 有 84% 的时间花在每行固定开销上，而它的行度数只有 6.9。
 *
 * 分组后这些开销按组摊：一次 Duplicate/Muls 覆盖整组，C 读入与写出各只要一组
 * 同步事件，非零元的整批 gather 也能跨行凑满槽位。上限 16 是 UB 与标量数组规模
 * 的折中；实际值由 InitLocalBuffers 按 UB 余量在 {16,8,4,2} 里自适应挑，凑不到 2
 * 就退回逐行路径(大 N 场景本来也不需要——固定开销已被大 N 摊薄)。 */
#define SPMM_ARCH22_ROW_GROUP_MAX 64
#define SPMM_ARCH22_ROW_GROUP_MIN 2

/* B 行槽位的半区数，用于分组路径的软件流水（见 kernels/spmm_kernel_group.h 的
 * RunPipedSegment）。
 *
 * 取 3 而不是 2 的理由是等待链的位置：一组填槽前必须挡住"上一次读这个半区"的
 * Axpy。两个半区时那次读正是紧邻的上一组，于是每组都要把 PIPE_V 排空一次，
 * MTE2 与 V 完全串行——现在的 AccumulateGroup 就是这个形态（一批 DMA、一个
 * MTE2_V 等齐、一批 Axpy、一个 V_MTE2 防覆写）。三个半区时挡的是上上一组，
 * 它在发本组 DMA 时早已做完，等待为零。
 *
 * 半区必须配轮换的事件 ID，而轮换必须用 TPipe::AllocEventID 而不是
 * FetchEventID：后者只返回"当前第一个空闲位"且不置占用位（见 CANN 的
 * kernel_tpipe_impl.h，AllocEventID 会 sbitset1、FetchEventID 不会），
 * 连续两次调用拿到同一个 ID，轮换退化成单 ID。同一 ID 上连着置位两次而中间
 * 没人等，旗只是一个 bit，会吞掉一个令牌。
 * DAV-2201 的 QUE_MAX_EVENT 是 8，3 + 2 个 ID 加上 bQueue/outQueue 自己占的
 * 那几个仍在池内。 */
#define SPMM_ARCH22_PIPE_STAGES 3

/* kernel 值类型选择器，供 host 端 spmm_arch22_kernel_launch 调用时按
 * aclDataType 映射后传入，驱动 kernels/spmm_kernel_dispatch.cpp 里的运行时分发。
 * 计算精度：无论选中哪个值类型，kernel 内部累加(reduce)恒为 float，
 * 满足任务书"fp16/bf16 用 fp32 累加"的精度要求；SPMM_VAL_FLOAT 是恒等情形。 */
#define SPMM_VAL_FLOAT     0
#define SPMM_VAL_HALF      1
#define SPMM_VAL_BF16      2
#define SPMM_VAL_COMPLEX64 3

#ifdef HOST_SPMM

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

inline struct aclsparseSpMatDescr *SpmmToMatInner(aclsparseConstSpMatDescr_t desc)
{
    return const_cast<struct aclsparseSpMatDescr *>(desc);
}

inline const struct aclsparseDnMatDescr *SpmmToDnMatInner(aclsparseConstDnMatDescr_t desc)
{
    return desc;
}

inline struct aclsparseDnMatDescr *SpmmToDnMatInner(aclsparseDnMatDescr_t desc)
{
    return desc;
}

#endif // HOST_SPMM

/* complex64 每个 chunk 的复数元素数上限。
 *
 * 这个值原本锁在 1024，理由是当时的 kernel 用 GatherMask 做 AoS->SoA 反交织，
 * 而 GatherMask 在生产算子 grid_sample 里验证过的单次调用规模只到 mask=1024。
 * 现在 kernel 已改成全程在 AoS 交织域上计算(见 kernels/
 * spmm_kernel_complex64.cpp 的 J 算子说明)，完全不再调用 GatherMask，这条
 * 限制的原始理由不再成立。
 *
 * 放宽到 2048 的直接收益：官方性能用例里有多个 N 落在 1024 与 2048 之间
 * (N=1031/1122/1200 等)，在 1024 上限下它们要跑两个 chunk，第二个 chunk 只处理
 * 7~176 列却要把整行的非零元循环(含每个非零元一次 B 行 DMA)完整重跑一遍，
 * 属于纯浪费；放宽后这些 case 单 chunk 即可覆盖。
 *
 * UB 安全性：新布局是 16 个 float/复数元素(见下方 SpmmArch22ComputeNChunksComplex64
 * 的注释)，2048 个元素占 2048*16*4 = 128KB，仍在 availUB 之内(约 184KB)，
 * 且 nMax 公式算出的上界(2944)大于本上限，因此 host 与 device 两侧都会被这个
 * 上限截断到同一个值，chunk 边界不会错位。*/
#define SPMM_ARCH22_COMPLEX64_MAX_CHUNK 2048

#define ROUNDUP(a, b) (((a) + (b) - 1) / (b) * (b))

/* tiling 数据结构，host 端填充后拷贝到 device 供 kernel 读取。
 * alphaImHost/betaImHost 仅 complex64 使用，其余 dtype 恒为 0。
 * idxBaseHost: CSR 索引基（0 或 1），kernel 读 rowOffsets/colIndices 时减去此值
 * 换算为数组下标；仅当 idxBaseHost==1 时代表任务书要求的 idxBase=1 场景。
 * rowBlockHost: host 实际使用的装箱粒度，0 表示逐行装箱。kernel 只有在这个值
 * 等于自己的 rowGroup 时才能推断"一组的行号连续"，见 ProcessChunkGrouped。
 * pipeModeHost: 分组路径是否走软件流水实现（1 走流水、0 走原路径）。由 host 端
 * 按环境变量与 UB 余量共同决定后写在这里，kernel 不自行推断——两侧必须看到同一个
 * 判定，否则 host 的装箱粒度与 kernel 的 rowGroup 会取自不同的预算函数，
 * "一组行号连续"的判定就会落空。 */
typedef struct {
     uint32_t M;
     uint32_t K;
     uint32_t N;
     uint32_t nnz;
     uint32_t blockDim;
     uint32_t nChunks;
     float    alphaHost;
     float    betaHost;
     int64_t  reorderOffset;
     int64_t  binEdgeOffset;
     int64_t  transposeRowOffset;
     int64_t  transposeSourceRowOffset;
     int64_t  transposePermutationOffset;
     float    alphaImHost;
     float    betaImHost;
     int32_t  idxBaseHost;
     int32_t  rowBlockHost;
     int32_t  pipeModeHost;
     uint32_t bRowsHost;
     uint32_t bColsHost;
     uint32_t aRowsHost;
     uint32_t aColsHost;
     int32_t  ldbHost;
     int32_t  ldcHost;
     int32_t  bOrderHost;
     int32_t  cOrderHost;
     int32_t  opAHost;
     int32_t  opBHost;
     int32_t  algHost;          // kernel 仅用它识别 fp32 high-precision 模式
     int32_t  genericPathHost;
     int32_t  vecRowOffsetHost;  // 流水路径是否把 B 行偏移 (col-base)*N 整段向量化预算
                                 // （int32），host 侧仅在 K*N<=INT32_MAX 且 nnz 够大时置 1
     int32_t  numericGuardHost;  // complex64 快路径是否启用行级非有限值回退：host 预检
                                 // |alpha|*max|A|*max|B| 逼近上界时置 1，避免拆分累加溢出
} SpmmArch22TilingData;

static inline uint64_t SpmmArch22WorkspaceSize(uint32_t M, uint32_t blockDim)
{
    uint64_t headerSize = 64;
    uint64_t tilingSize = ROUNDUP(sizeof(SpmmArch22TilingData), 64);
    uint64_t reorderSize = ROUNDUP((uint64_t)M * sizeof(int32_t), 64);
    uint64_t binEdgeSize = ROUNDUP(((uint64_t)blockDim + 1) * sizeof(int32_t), 64);
    return headerSize + tilingSize + reorderSize + binEdgeSize;
}

/* 单个 chunk 的 UB 字节/元素。逐项对应 kernel 侧 InitLocalBuffers 的五次
 * InitBuffer，不写成合并后的常数。此前这里是 (BUFFER_NUM+1)*valBytes +
 * 2*sizeof(float)，把 half/bf16 的 accBuf 与 bFloatBuf 只算了一份：实际
 * 18 字节/元素却按 14 估算。device 侧当时用的是同一个错误口径，所以两边 chunk
 * 边界仍然一致、不会错位，但 nMax 被高估到 13456，N 超过 10464 时 kernel 的
 * InitBuffer 会向 192KB 的 UB 申请约 236KB。
 * 两侧必须逐项保持一致，任何一侧改动都要同步另一侧。 */
SPMM_ARCH22_HD uint64_t SpmmArch22BytesPerElem(uint32_t valBytes, uint32_t depth)
{
    return (uint64_t)depth * valBytes +                          /* bQueue 预取环 */
           valBytes +                                            /* outQueue */
           sizeof(float) +                                       /* tmpBuf */
           ((valBytes == sizeof(float)) ? 0 : 2 * sizeof(float)); /* accBuf+bFloatBuf */
}

/* 单个 chunk 的元素数要按哪个粒度对齐。
 *
 * SPMM_ARCH22_ALIGN 是元素数口径，只有 4 字节类型下才恰好等于 32 字节。窄类型
 * (fp16/bf16，2 字节)按 8 个元素对齐只有 16 字节，于是 batch[j * nAlignChunk]、
 * stage[g * nAlignChunk] 这类槽位偏移会落在非 32 字节边界上，DataCopyPad 在
 * vector core 上抛 ADDR_MISALIGN（精度泛化 1000 例里 fp16/bf16 各 250 例中崩了
 * 248 例，首个崩溃用例 31x17x1 算出的 nAlignChunk 正好是 8，即 16 字节）。这里
 * 按字节反推元素数，任何 dtype 下槽位步长都是 32 的整数倍。
 *
 * 只改对齐粒度、不改 nMax 的上界来源，所以 SpmmArch22ComputeNChunks 的返回值逐值
 * 不变(fp16/bf16 的 nMax=10240 本就是 16 的倍数，fp32 的粒度仍是 8)，chunk 边界
 * 不会错位。 */
SPMM_ARCH22_HD uint32_t SpmmArch22AlignElems(uint32_t valBytes)
{
    if (valBytes == 0 || valBytes >= SPMM_ARCH22_UB_ADDR_ALIGN) {
        return SPMM_ARCH22_ALIGN;
    }
    uint32_t byAddr = SPMM_ARCH22_UB_ADDR_ALIGN / valBytes;
    return byAddr > SPMM_ARCH22_ALIGN ? byAddr : SPMM_ARCH22_ALIGN;
}

SPMM_ARCH22_HD uint32_t SpmmArch22NMaxForDepth(uint32_t valBytes, uint32_t depth)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint64_t bytesPerElem = SpmmArch22BytesPerElem(valBytes, depth);
    /* BytesPerElem always adds sizeof(float) unconditionally, so it can never
     * actually be 0 -- but that fact lives in the callee, not here, so the
     * static check can't see it. Same reasoning as SpmmArch22ComputeNChunks
     * just below. */
    if (bytesPerElem == 0) {
        bytesPerElem = sizeof(float);
    }
    uint64_t nMax = availUB / bytesPerElem;
    uint32_t align = SpmmArch22AlignElems(valBytes);
    if (align == 0) {
        align = SPMM_ARCH22_ALIGN;
    }
    nMax = (nMax / align) * align;
    if (nMax == 0) {
        nMax = align;
    }
    return (uint32_t)nMax;
}

/* 选 bQueue 的预取深度：取能让整个 N 仍然装进单个 chunk 的最大深度，装不下就退回
 * SPMM_ARCH22_BUFFER_NUM。
 *
 * 这条规则保证 SpmmArch22ComputeNChunks 的返回值与加深预取之前逐值相同——深度只
 * 在"不会多切出一个 chunk"的前提下才加大，所以 chunk 边界不会错位，大 N 场景
 * 自动退回原来的深度 2。大 N 本来也不需要深预取：N 越大单次 B 行搬运越接近
 * 连续大块传输，发起延迟占比可忽略，瓶颈回到带宽。
 *
 * host 与 device 都调用本函数，两侧看到的 N 和 valBytes 一致，因此选出的深度必然
 * 一致。 */
SPMM_ARCH22_HD uint32_t SpmmArch22PrefetchDepth(uint32_t N, uint32_t valBytes)
{
    uint32_t depth = SPMM_ARCH22_BQUEUE_MAX;
    while (depth > SPMM_ARCH22_BUFFER_NUM) {
        if (SpmmArch22NMaxForDepth(valBytes, depth) >= N) {
            return depth;
        }
        depth >>= 1;
    }
    return SPMM_ARCH22_BUFFER_NUM;
}

/* 单个 chunk 对齐后的元素数，下面两个函数与 kernel 侧 InitLocalBuffers 共用。 */
SPMM_ARCH22_HD uint64_t SpmmArch22NAlignChunk(uint32_t N, uint32_t valBytes)
{
    uint32_t nMax = SpmmArch22NMaxForDepth(valBytes, SpmmArch22PrefetchDepth(N, valBytes));
    if (nMax > N) {
        nMax = N;
    }
    return ROUNDUP((uint64_t)nMax, SpmmArch22AlignElems(valBytes));
}

/* B 行槽位数与分组行数。两者都只花"按 N 伸缩的预算之外剩下的 UB"，所以不进
 * SpmmArch22BytesPerElem 的口径、不影响 nChunks；但 host 与 device 必须算出同一个
 * 值——host 用 SpmmArch22RowGroup 决定连续行块的粒度，kernel 用它决定一组几行，
 * 两者不等的话组边界就会跨过行块边界，kernel 里的连续性判定会一直落空。 */
SPMM_ARCH22_HD uint32_t SpmmArch22BatchSlots(uint32_t N, uint32_t valBytes)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint32_t depth = SpmmArch22PrefetchDepth(N, valBytes);
    uint64_t nAlign = SpmmArch22NAlignChunk(N, valBytes);
    uint64_t used = SpmmArch22BytesPerElem(valBytes, depth) * nAlign;
    uint64_t slotBytes = valBytes + ((valBytes == sizeof(float)) ? 0 : sizeof(float));
    for (uint32_t s = SPMM_ARCH22_BATCH_SLOTS_MAX; s >= SPMM_ARCH22_BATCH_SLOTS_MIN; s >>= 1) {
        if (used + (uint64_t)s * nAlign * slotBytes <= availUB) {
            return s;
        }
    }
    return 0;
}

SPMM_ARCH22_HD uint32_t SpmmArch22RowGroup(uint32_t N, uint32_t valBytes)
{
    uint32_t slots = SpmmArch22BatchSlots(N, valBytes);
    if (slots == 0) {
        return 0;
    }
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint32_t depth = SpmmArch22PrefetchDepth(N, valBytes);
    uint64_t nAlign = SpmmArch22NAlignChunk(N, valBytes);
    uint64_t used = SpmmArch22BytesPerElem(valBytes, depth) * nAlign;
    uint64_t slotBytes = valBytes + ((valBytes == sizeof(float)) ? 0 : sizeof(float));
    uint64_t slotUsed = (uint64_t)slots * nAlign * slotBytes;
    uint64_t groupBytes = sizeof(float) + ((valBytes == sizeof(float)) ? 0 : valBytes);
    for (uint32_t g = SPMM_ARCH22_ROW_GROUP_MAX; g >= SPMM_ARCH22_ROW_GROUP_MIN; g >>= 1) {
        if (used + slotUsed + (uint64_t)g * nAlign * groupBytes <= availUB) {
            return g;
        }
    }
    return 0;
}

/* 流水模式下的槽位数与分组行数。与上面那对函数唯一的差别是每个槽位要
 * SPMM_ARCH22_PIPE_STAGES 份 ValT 而不是一份——半区是同一块 UB 里首尾相接的
 * 三段，窄类型的 fp32 Cast 暂存仍只要一份（它在组内被 Cast 写、被 Axpy 读，
 * PIPE_V 内部有序，不需要轮换）。
 *
 * 同样只花"按 N 伸缩的预算之外剩下的 UB"，因此不进 SpmmArch22BytesPerElem 的
 * 口径、不影响 nChunks——两种模式下的列分块边界逐值相同，切换开关不会让 host
 * 与 device 对 chunk 的理解漂移。
 *
 * 列索引与非零值的段缓存不在这里计：它们是与 N 无关的定长 tile
 * (SPMM_ARCH22_NZ_TILE)，已经算在 SPMM_ARCH22_UB_OVERHEAD 预留的那 4KB 里，
 * 与逐行整批路径复用同一组缓冲。 */
SPMM_ARCH22_HD uint64_t SpmmArch22PipeSlotBytes(uint32_t valBytes)
{
    return (uint64_t)SPMM_ARCH22_PIPE_STAGES * valBytes +
           ((valBytes == sizeof(float)) ? 0 : sizeof(float));
}

SPMM_ARCH22_HD uint32_t SpmmArch22PipeSlots(uint32_t N, uint32_t valBytes)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint32_t depth = SpmmArch22PrefetchDepth(N, valBytes);
    uint64_t nAlign = SpmmArch22NAlignChunk(N, valBytes);
    uint64_t used = SpmmArch22BytesPerElem(valBytes, depth) * nAlign;
    uint64_t slotBytes = SpmmArch22PipeSlotBytes(valBytes);
    for (uint32_t s = SPMM_ARCH22_BATCH_SLOTS_MAX; s >= SPMM_ARCH22_BATCH_SLOTS_MIN; s >>= 1) {
        if (used + (uint64_t)s * nAlign * slotBytes <= availUB) {
            return s;
        }
    }
    return 0;
}

SPMM_ARCH22_HD uint32_t SpmmArch22PipeRowGroup(uint32_t N, uint32_t valBytes)
{
    uint32_t slots = SpmmArch22PipeSlots(N, valBytes);
    if (slots == 0) {
        return 0;
    }
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint32_t depth = SpmmArch22PrefetchDepth(N, valBytes);
    uint64_t nAlign = SpmmArch22NAlignChunk(N, valBytes);
    uint64_t used = SpmmArch22BytesPerElem(valBytes, depth) * nAlign;
    uint64_t slotUsed = (uint64_t)slots * nAlign * SpmmArch22PipeSlotBytes(valBytes);
    uint64_t groupBytes = sizeof(float) + ((valBytes == sizeof(float)) ? 0 : valBytes);
    for (uint32_t g = SPMM_ARCH22_ROW_GROUP_MAX; g >= SPMM_ARCH22_ROW_GROUP_MIN; g >>= 1) {
        if (used + slotUsed + (uint64_t)g * nAlign * groupBytes <= availUB) {
            return g;
        }
    }
    return 0;
}

/* 实际生效的装箱粒度：块数太少时按块装箱会失去均衡能力（极端情况每核不到一块），
 * 退回逐行。阈值取 4 倍核数：LPT 在每核 4 个以上的块时已能把负载差压到一块以内。
 * host 装箱与 tiling 都走这个函数，kernel 靠 tiling 里的结果判断能不能信任连续性，
 * 三处必须同源——装箱走了逐行而 tiling 仍写着块粒度的话，kernel 会把打散的行号
 * 当成连续区间去取 rowOffsets，算出来的结果是错的。 */
SPMM_ARCH22_HD uint32_t SpmmArch22EffectiveRowBlock(uint32_t M, uint32_t blockDim,
    uint32_t rowBlock)
{
    if (rowBlock < 2 || blockDim == 0) {
        return 0;
    }
    uint64_t nBlocks = ((uint64_t)M + rowBlock - 1) / rowBlock;
    return (nBlocks >= (uint64_t)blockDim * 4) ? rowBlock : 0;
}

/* N chunk 数量与 kernel 侧的 UB 预算耦合：values/matB/matC 数据类型越窄(half/bf16)，
 * 单个 chunk 能塞进 UB 的元素越多，chunk 数越少。 */
SPMM_ARCH22_HD uint32_t SpmmArch22ComputeNChunks(uint32_t N, uint32_t valBytes)
{
    uint32_t nMax = SpmmArch22NMaxForDepth(valBytes, SpmmArch22PrefetchDepth(N, valBytes));
    /* NMaxForDepth 的返回值已经不会小于 SPMM_ARCH22_ALIGN，这里再兜一次是为了让
     * 除数非零在本函数内可见——静态检查看不穿被调函数里的钳位。 */
    if (nMax == 0) {
        nMax = SPMM_ARCH22_ALIGN;
    }
    return (uint32_t)((N + nMax - 1) / nMax);
}

/* complex64 每个 chunk 的 UB 预算(float 单位，每个复数元素占 2 个 float 的存储位)。
 * 全 AoS 交织布局，每个 buffer 都是 2*nAlignChunk 个 float 宽:
 * raw 双缓冲=2*2=4，outQueue 单份=2，两个累加器 accA/accB=2+2，J 算子的中间
 * 结果 swapped=2，符号表 sign=2，配对下标表 pairOffset(uint32_t，同宽)=2
 * => 16 个 float/元素，与 kernels/spmm_kernel_complex64.cpp 的 Init() 保持
 * 同一口径。 */
#define SPMM_ARCH22_C64_FLOATS_PER_ELEM 16

/* 单个 chunk 能容纳的复数元素数，不按 N 截断。SPMM_ARCH22_COMPLEX64_MAX_CHUNK
 * 远小于 UB 余量算出的上界，实际恒被它截断。钳到 SPMM_ARCH22_ALIGN 保证非零，
 * 调用方可以直接拿它做除数。 */
SPMM_ARCH22_HD uint32_t SpmmArch22NMaxChunkComplex64Raw(void)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint64_t nMax = availUB / (SPMM_ARCH22_C64_FLOATS_PER_ELEM * sizeof(float));
    nMax = (nMax / SPMM_ARCH22_ALIGN) * SPMM_ARCH22_ALIGN;
    if (nMax > SPMM_ARCH22_COMPLEX64_MAX_CHUNK) {
        nMax = SPMM_ARCH22_COMPLEX64_MAX_CHUNK;
    }
    if (nMax == 0) {
        nMax = SPMM_ARCH22_ALIGN;
    }
    return (uint32_t)nMax;
}

/* 同上，再按 N 截断——N 比一个 chunk 还窄时只按 N 开缓冲。kernel 侧 Init() 用的
 * 是这一支，host 与 device 必须走同一个函数，否则 chunk 边界会错位。 */
SPMM_ARCH22_HD uint32_t SpmmArch22NMaxChunkComplex64(uint32_t N)
{
    uint32_t nMax = SpmmArch22NMaxChunkComplex64Raw();
    return (nMax > N) ? N : nMax;
}

static inline uint32_t SpmmArch22ComputeNChunksComplex64(uint32_t N)
{
    uint32_t nMax = SpmmArch22NMaxChunkComplex64Raw();
    /* 同 SpmmArch22ComputeNChunks：Raw() 的返回值已不会小于 SPMM_ARCH22_ALIGN，
     * 这里再兜一次是为了让除数非零在本函数内可见。 */
    if (nMax == 0) {
        nMax = SPMM_ARCH22_ALIGN;
    }
    return (uint32_t)((N + nMax - 1) / nMax);
}

/* complex64 版的槽位数与分组行数，口径对应 kernels/spmm_kernel_complex64.cpp 的
 * InitLocalBuffers：全 AoS 交织，每行 buffer 宽 2*nAlignChunk 个 float，一组行要
 * accA/accB 两份。与实数版同理，host 与 device 必须算出同一个分组行数。 */
SPMM_ARCH22_HD uint64_t SpmmArch22NAlignChunkComplex64(uint32_t N)
{
    return ROUNDUP(SpmmArch22NMaxChunkComplex64(N), SPMM_ARCH22_ALIGN);
}

SPMM_ARCH22_HD uint32_t SpmmArch22BatchSlotsComplex64(uint32_t N)
{
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint64_t nAlign = SpmmArch22NAlignChunkComplex64(N);
    uint64_t aosBytes = nAlign * 2 * sizeof(float);
    uint64_t used = SPMM_ARCH22_C64_FLOATS_PER_ELEM * nAlign * sizeof(float);
    for (uint32_t s = SPMM_ARCH22_BATCH_SLOTS_MAX; s >= SPMM_ARCH22_BATCH_SLOTS_MIN; s >>= 1) {
        if (used + (uint64_t)s * aosBytes <= availUB) {
            return s;
        }
    }
    return 0;
}

SPMM_ARCH22_HD uint32_t SpmmArch22RowGroupComplex64(uint32_t N)
{
    uint32_t slots = SpmmArch22BatchSlotsComplex64(N);
    if (slots == 0) {
        return 0;
    }
    uint64_t availUB = SPMM_ARCH22_UB_SIZE - SPMM_ARCH22_UB_OVERHEAD;
    uint64_t nAlign = SpmmArch22NAlignChunkComplex64(N);
    uint64_t aosBytes = nAlign * 2 * sizeof(float);
    uint64_t used =
        SPMM_ARCH22_C64_FLOATS_PER_ELEM * nAlign * sizeof(float) + (uint64_t)slots * aosBytes;
    for (uint32_t g = SPMM_ARCH22_ROW_GROUP_MAX; g >= SPMM_ARCH22_ROW_GROUP_MIN; g >>= 1) {
        /* 一组的行数不能超过槽位数：complex64 的 InitGroupAcc 把整组的 C 借批槽
         * 搬入(省一块与分组同宽的 UB)，写的是 batch[g * stride]。
         * SPMM_ARCH22_ROW_GROUP_MAX(64) 大于 SPMM_ARCH22_BATCH_SLOTS_MAX(32)，而
         * g 取到 64 需要 1344 * nAlign <= availUB，即 nAlign <= 136；slots 在
         * nAlign <= 576 内都是 32。两段一交，N <= 136 时 g=64 > slots=32，末尾那
         * 32 行的搬运写出 bBatchBuf、落进紧邻分配的 accAGroupBuf——原始 input 没乘
         * beta 就进了累加器实部。精度泛化的 250 例 complex64 里 52 例失败，n 全部
         * 落在该区间内(最大 131)，区间外 117 例无一失败；且错误只出现在实部，与该
         * 越界的签名一致。区间内另有 81 例通过，是那些数据/形状没让越界显形。 */
        if (g > slots) {
            continue;
        }
        if (used + (uint64_t)g * 2 * aosBytes <= availUB) {
            return g;
        }
    }
    return 0;
}

/* host 侧 kernel launch 入口，定义于 kernels/spmm_kernel_dispatch.cpp，供
 * spmm_host.cpp 调用；_out 版本把 bias 输入 matC 与输出 matY 分开。valType
 * 取值见上面的 SPMM_VAL_* 宏。 */
void spmm_arch22_kernel_launch(
    void *rowOff, void *colIdx, void *values,
    void *matB, void *matC, void *workspace, void *tiling,
    uint32_t blockDim, void *stream, int32_t valType);

void spmm_arch22_kernel_launch_out(
    void *rowOff, void *colIdx, void *values,
    void *matB, void *matC, void *matY, void *workspace, void *tiling,
    uint32_t blockDim, void *stream, int32_t valType);

#endif
