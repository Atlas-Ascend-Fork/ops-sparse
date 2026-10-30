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
 * \file prune_check_kernel.cpp
 * \brief aclsparseLtSpMMAPruneCheck 的 Device 侧实现（arch35 / RegBase / AIV Vector）。
 *
 * 2:4 结构化稀疏条件校验：逐分组非零计数 + 跨分组布尔归约 → 单 int 标量 d_valid。
 *
 * 批量语义：tile 网格展开为 batch × 行带 × 列块，外层 batch 循环 + 内层
 * tile-in-matrix grid-stride 领取（batchIdx 每批一次，行带/列块坐标计数器
 * 增量推进，热循环零除法/取模），batch 基址偏移 batchStride 元素；d_valid =
 * OR(全部 batch 的判定)——单 batch 判定链不变，多写者同值天然支持跨 batch
 * 的 OR 归约。numBatches==1 时 batchIdx 恒 0，与单 batch 路径完全一致。
 *
 * dtype 处理（R8 硬约束）：全程按等宽无符号整数【位视图】做位级判零
 * （FP32→uint32_t，FP16/BF16→uint16_t，INT8→uint8_t），不做任何数据类型转换，
 * 禁止引入 BF16 static_cast / Cast（prune kernel 已踩过的 DAV-3510 后端陷阱）；
 * ±0 经幅值掩码按零处理，NaN/非规格数按非零处理，与 golden 的 (bits & magMask) != 0
 * 位级等价。
 *
 * d_valid 多 block 归约：host 侧 aclrtMemsetAsync 预清零 + 仅违规 block 以
 * GlobalTensor::SetValue 写 1（多写者写同一值，无 last-writer 歧义，结果确定）。
 */

#include <cstdint>

#include "kernel_operator.h"
#include "prune_check_kernel.h"

namespace {
constexpr uint32_t kVectorBytes = 256;

// g=2 分组（FP32 的 1:2 模式）：nnz(v0,v1) > keep(=1) → violVal=1。
// 每条 lane 对应一个分组；输入为组内位置 0/1 的元素向量。
// 位视图判零：(bits & magMask) != 0，全程等宽无符号位运算，无转换链（R8）。
// keep 为编译期常量（g/2），计数超限用向量-标量比较（仓内 Compares 已验证路径）。
template <typename U>
__simd_callee__ inline void GroupViolation2(AscendC::Reg::RegTensor<U>& v0,
    AscendC::Reg::RegTensor<U>& v1, AscendC::Reg::RegTensor<U>& magMask,
    AscendC::Reg::RegTensor<U>& one, AscendC::Reg::RegTensor<U>& zero,
    AscendC::Reg::RegTensor<U>& violVal)
{
    auto mask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg nzMask0, nzMask1, violMask;
    AscendC::Reg::RegTensor<U> mag0, mag1, nz0, nz1, nnz;
    AscendC::Reg::And<U>(mag0, v0, magMask, mask);
    AscendC::Reg::And<U>(mag1, v1, magMask, mask);
    AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask0, mag0, static_cast<U>(0), mask);
    AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask1, mag1, static_cast<U>(0), mask);
    AscendC::Reg::Select<U>(nz0, one, zero, nzMask0);
    AscendC::Reg::Select<U>(nz1, one, zero, nzMask1);
    AscendC::Reg::Add<U>(nnz, nz0, nz1, mask);
    AscendC::Reg::Compares<U, AscendC::CMPMODE::GT>(violMask, nnz, static_cast<U>(1), mask);
    AscendC::Reg::Select<U>(violVal, one, zero, violMask);
}

// g=4 分组（FP16/BF16/INT8 的 2:4 模式）：nnz(v0..v3) > keep(=2) → violVal=1。
// INT8（1 字节）全字节判零，And 恒等省略（每 256B 省 4 条向量指令，O1 指令计数依据）。
template <typename U>
__simd_callee__ inline void GroupViolation4(AscendC::Reg::RegTensor<U>& v0,
    AscendC::Reg::RegTensor<U>& v1, AscendC::Reg::RegTensor<U>& v2,
    AscendC::Reg::RegTensor<U>& v3,
    AscendC::Reg::RegTensor<U>& magMask, AscendC::Reg::RegTensor<U>& one,
    AscendC::Reg::RegTensor<U>& zero, AscendC::Reg::RegTensor<U>& violVal)
{
    auto mask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg nzMask0, nzMask1, nzMask2, nzMask3, violMask;
    AscendC::Reg::RegTensor<U> nz0, nz1, nz2, nz3, nnz;
    if constexpr (sizeof(U) == 1) {
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask0, v0, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask1, v1, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask2, v2, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask3, v3, static_cast<U>(0), mask);
    } else {
        AscendC::Reg::RegTensor<U> mag0, mag1, mag2, mag3;
        AscendC::Reg::And<U>(mag0, v0, magMask, mask);
        AscendC::Reg::And<U>(mag1, v1, magMask, mask);
        AscendC::Reg::And<U>(mag2, v2, magMask, mask);
        AscendC::Reg::And<U>(mag3, v3, magMask, mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask0, mag0, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask1, mag1, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask2, mag2, static_cast<U>(0), mask);
        AscendC::Reg::Compares<U, AscendC::CMPMODE::NE>(nzMask3, mag3, static_cast<U>(0), mask);
    }
    AscendC::Reg::Select<U>(nz0, one, zero, nzMask0);
    AscendC::Reg::Select<U>(nz1, one, zero, nzMask1);
    AscendC::Reg::Select<U>(nz2, one, zero, nzMask2);
    AscendC::Reg::Select<U>(nz3, one, zero, nzMask3);
    AscendC::Reg::Add<U>(nnz, nz0, nz1, mask);
    AscendC::Reg::Add<U>(nnz, nnz, nz2, mask);
    AscendC::Reg::Add<U>(nnz, nnz, nz3, mask);
    // keep 为编译期常量（g/2），计数超限用向量-标量比较（仓内 Compares 已验证路径）。
    AscendC::Reg::Compares<U, AscendC::CMPMODE::GT>(violMask, nnz, static_cast<U>(2), mask);
    AscendC::Reg::Select<U>(violVal, one, zero, violMask);
}

// 列尾块条件预清零（O2）：仅 validCols < tileCols 的块执行（见 CopyIn 调用处）。
// 清零保证 slot 尾部 stale 区为 0 → 补零组 nnz=0，不产生假违规（尾块正确性载体）。
template <typename U>
__simd_vf__ inline void PruneCheckClearInput(__ubuf__ U* input, uint32_t elementCount)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    auto mask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<U> zero;
    AscendC::Reg::Duplicate<U>(zero, static_cast<U>(0), mask);
    for (uint32_t offset = 0; offset < elementCount; offset += lanes) {
        AscendC::Reg::StoreAlign<U>(input + offset, zero, mask);
    }
}

// RowScan 判定链（O1 必落地）：LoadAlign → DeInterleave（每 lane=一组）
// → And → Compares(NE) → Select → Add → Compares(GT) → Select → Or。
// 循环不变常量（one/zero/magMask）每 VF 调用 Duplicate 一次，不进 chunk 循环。
//
// DeInterleave(src, src) 的自拼接语义使每 256B 块仅前 lanes/groupSize 个 lane
// 是有效组视图（spmma_compress 同款约束——其 StoreMetadata 仅消费
// groupCount = validCount/groupSize 个前段 lane），后段 lane 为越界混叠视图。
// 因此累或（Or）使用前缀子集掩码（VL32/VL64 按类型分派）将后段 lane 排除在
// 违规归约之外，anyViolate 后段恒为 0 —— acc 全量扫描不会引入假阳性，
// 且对 StoreAlign 的 lane→内存映射完全鲁棒。
template <typename U>
__simd_vf__ inline void PruneCheckRowGroups(__ubuf__ U* input, __ubuf__ U* acc,
    uint32_t pitchElems, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    constexpr uint32_t groupSize = (sizeof(U) == 4) ? 2 : 4;
    constexpr uint32_t groupLanes = lanes / groupSize;
    auto mask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
    // 前缀子集掩码（循环不变量）：仅前 groupLanes 个 lane 参与累或。
    AscendC::Reg::MaskReg groupMask;
    if constexpr (groupLanes <= 32) {
        groupMask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::VL32>();
    } else {
        groupMask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::VL64>();
    }
    AscendC::Reg::RegTensor<U> magMask, one, zero, anyViolate, violVal;
    // 判零幅值掩码（constexpr 局部常量，spmma_compress SelectTwoAndStore 同款）：
    // 屏蔽符号位后判零 → ±0 按零、NaN/非规格数按非零；
    // INT8（1 字节整型）无符号位，掩码 0xff 恒等（GroupViolation4 中省略 And）。
    constexpr U magMaskValue = (sizeof(U) == 4) ? static_cast<U>(0x7fffffffU) :
        (sizeof(U) == 2) ? static_cast<U>(0x7fffU) : static_cast<U>(0xffU);
    AscendC::Reg::Duplicate<U>(magMask, magMaskValue, mask);
    AscendC::Reg::Duplicate<U>(one, static_cast<U>(1), mask);
    AscendC::Reg::Duplicate<U>(zero, static_cast<U>(0), mask);
    AscendC::Reg::Duplicate<U>(anyViolate, static_cast<U>(0), mask);
    for (uint32_t row = 0; row < validRows; ++row) {
        __ubuf__ U* rowBase = input + row * pitchElems;
        // off 为 lanes 倍数 → LoadAlign 源地址 256B 对齐；尾块越界读落在已清零区。
        for (uint32_t offset = 0; offset < validCols; offset += lanes) {
            AscendC::Reg::RegTensor<U> src, even, odd, v0, v1, v2, v3;
            AscendC::Reg::LoadAlign<U>(src, rowBase + offset);
            if constexpr (groupSize == 2) {
                // FP32（1:2）：单次 DeInterleave 拆出组内两个位置。
                AscendC::Reg::DeInterleave<U>(v0, v1, src, src);
                GroupViolation2<U>(v0, v1, magMask, one, zero, violVal);
            } else {
                // FP16/BF16/INT8（2:4）：三级 DeInterleave 拆出组内四个位置。
                AscendC::Reg::DeInterleave<U>(even, odd, src, src);
                AscendC::Reg::DeInterleave<U>(v0, v2, even, even);
                AscendC::Reg::DeInterleave<U>(v1, v3, odd, odd);
                GroupViolation4<U>(v0, v1, v2, v3, magMask, one, zero, violVal);
            }
            // Reg::Or dst==src0 别名累或（硬件读后写语义）。
            AscendC::Reg::Or<U>(anyViolate, anyViolate, violVal, groupMask);
        }
    }
    AscendC::Reg::StoreAlign<U>(acc, anyViolate, mask);
}

// ColScan 变体（O1 同款常量提升）：同列相邻 groupSize 行拼组（每 lane=一列的一个分组），
// 取数几何与 spmma_compress CompressColumns 一致（LoadAlign×groupSize）。
// validRows 为 groupSize 的倍数（行带尾由维度对齐保证），垂直组不跨行带边界。
template <typename U>
__simd_vf__ inline void PruneCheckColGroups(__ubuf__ U* input, __ubuf__ U* acc,
    uint32_t pitchElems, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    constexpr uint32_t groupSize = (sizeof(U) == 4) ? 2 : 4;
    auto mask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<U> magMask, one, zero, anyViolate, violVal;
    // 判零幅值掩码（constexpr 局部常量）：屏蔽符号位后判零 → ±0 按零、
    // NaN/非规格数按非零；INT8（1 字节）无符号位，掩码 0xff 恒等。
    constexpr U magMaskValue = (sizeof(U) == 4) ? static_cast<U>(0x7fffffffU) :
        (sizeof(U) == 2) ? static_cast<U>(0x7fffU) : static_cast<U>(0xffU);
    AscendC::Reg::Duplicate<U>(magMask, magMaskValue, mask);
    AscendC::Reg::Duplicate<U>(one, static_cast<U>(1), mask);
    AscendC::Reg::Duplicate<U>(zero, static_cast<U>(0), mask);
    AscendC::Reg::Duplicate<U>(anyViolate, static_cast<U>(0), mask);
    for (uint32_t offset = 0; offset < validCols; offset += lanes) {
        for (uint32_t row = 0; row < validRows; row += groupSize) {
            AscendC::Reg::RegTensor<U> v0, v1, v2, v3;
            AscendC::Reg::LoadAlign<U>(v0, input + row * pitchElems + offset);
            AscendC::Reg::LoadAlign<U>(v1, input + (row + 1) * pitchElems + offset);
            if constexpr (groupSize == 4) {
                AscendC::Reg::LoadAlign<U>(v2, input + (row + 2) * pitchElems + offset);
                AscendC::Reg::LoadAlign<U>(v3, input + (row + 3) * pitchElems + offset);
                GroupViolation4<U>(v0, v1, v2, v3, magMask, one, zero, violVal);
            } else {
                GroupViolation2<U>(v0, v1, magMask, one, zero, violVal);
            }
            // Reg::Or dst==src0 别名累或（同 RowScan）。
            AscendC::Reg::Or<U>(anyViolate, anyViolate, violVal, mask);
        }
    }
    AscendC::Reg::StoreAlign<U>(acc, anyViolate, mask);
}

template <AscendC::HardEvent Event>
__aicore__ inline void SynchronizePipeline()
{
    AscendC::SetFlag<Event>(EVENT_ID0);
    AscendC::WaitFlag<Event>(EVENT_ID0);
}

// 坐标与有效尺寸均按物理存储解释；末尾 tile 的有效尺寸可能小于分配尺寸。
// batch 为批量维度下标（d_valid = OR(全部 batch)，判定链不变）。
struct TilePosition {
    uint64_t batch;
    uint64_t rowOffset;
    uint64_t colOffset;
    uint32_t validRows;
    uint32_t validCols;
};

// 由行带/列块计数器直接换算 tile 坐标与有效尺寸（无除法/取模；计数器语义
// 与领取循环见 Process）。行带/列块计数在 host 保证下不越界：
// rowBand < ceil(rows/tileRows)、colChunk < ceil(cols/tileCols)。
__aicore__ inline TilePosition MakeTilePosition(const PruneCheckTilingData& td, uint64_t batch,
    uint64_t rowBand, uint64_t colChunk)
{
    TilePosition tile{};
    tile.batch = batch;
    tile.rowOffset = rowBand * td.tileRows;
    tile.colOffset = colChunk * td.tileCols;
    tile.validRows = static_cast<uint32_t>(td.rows - tile.rowOffset < td.tileRows ?
        td.rows - tile.rowOffset : td.tileRows);
    tile.validCols = static_cast<uint32_t>(td.cols - tile.colOffset < td.tileCols ?
        td.cols - tile.colOffset : td.tileCols);
    return tile;
}

template <typename U>
class PruneCheckKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR dValid,
        const PruneCheckTilingData& tiling, AscendC::TPipe& pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const TilePosition& tile);
    __aicore__ inline void Compute(const TilePosition& tile);

    PruneCheckTilingData tilingData_;
    // 输入 GM 只读语义说明：框架 DataCopyPad(GM→UB) 重载要求 dst LocalTensor<T>
    // 与 src GlobalTensor<T> 为同一元素类型（无 const 配对重载），GlobalTensor<const U>
    // 会使模板推导冲突（T=U vs T=const U）；故沿用可写元素类型（prune/spmma_compress
    // 家族同款），只读约束由实现保证（全程无 SetValue/写路径，仅 dValidGm_ 写出）。
    AscendC::GlobalTensor<U> inputGm_;
    // d_valid 共享单元：host 预清零 + 仅违规 block 写 1。
    AscendC::GlobalTensor<int32_t> dValidGm_;
    // UB 规划：inputBuf（tile 暂存，行 pitch 256B 对齐）+ reduceBuf（acc 落地槽 256B）。
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuffer_;
    AscendC::LocalTensor<U> inputLocal_;
    AscendC::LocalTensor<U> accLocal_;
};

template <typename U>
__aicore__ inline void PruneCheckKernel<U>::Init(GM_ADDR input, GM_ADDR dValid,
    const PruneCheckTilingData& tiling, AscendC::TPipe& pipe)
{
    tilingData_ = tiling;
    // 全量跨度含 batch 项（host #11 已校验无溢出，uint64 乘加安全）。
    const uint64_t spanElems = (tilingData_.numBatches - 1) * tilingData_.batchStride +
        (tilingData_.rows - 1) * tilingData_.ld + tilingData_.cols;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(input), spanElems);
    dValidGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dValid), 1);
    pipe.InitBuffer(inputBuffer_, tilingData_.inputBufferBytes);
    pipe.InitBuffer(reduceBuffer_, kVectorBytes);
    inputLocal_ = inputBuffer_.Get<U>();
    accLocal_ = reduceBuffer_.Get<U>();
}

template <typename U>
__aicore__ inline void PruneCheckKernel<U>::Process()
{
    constexpr uint32_t lanes = kVectorBytes / sizeof(U);
    bool blockViolation = false;
    // grid-stride 领取 tile（各 tile 相互独立，组级校验无跨 tile 依赖，无核间通信）：
    // 外层 batch 循环 + 内层 tile-in-matrix grid-stride，batchIdx 每批一次；
    // 行带/列块二维坐标用计数器增量推进（tim 每步 +stride → colChunk +=
    // stride%tPRB、rowBand += stride/tPRB + 列进位，除法仅预计算/初始化每
    // kernel O(1) 次），热循环零除法/取模（Ascend 标量单元除法代价高）。
    // 覆盖集合与原「线性 tileId = blockIdx + k·blockNum」grid-stride 一致：
    // 每个 (batch, tileInMatrix) 恰被一个 block 领取；tPM < blockNum 时
    // blockIdx ≥ tPM 的 block 空转（numBatches==1 时 host blockDim ≤ totalTiles
    // = tPM 保证无空转；多 batch 小矩阵空转为已知限制，见白盒覆盖说明 §5）。
    //
    // acc 全量扫描：RowScan 的混叠段 lane 已被 groupMask 压为恒 0（见
    // PruneCheckRowGroups），ColScan 全 lane 有效（越界列已清零），
    // 因此 acc 任意非零都来自有效组视图的违规信号。
    const uint64_t stride = AscendC::GetBlockNum();
    const uint64_t blockIdx = AscendC::GetBlockIdx();
    const uint64_t tpm = tilingData_.tilesPerMatrix;
    const uint64_t tprb = tilingData_.tilesPerRowBand;
    const uint64_t colStep = stride % tprb;  // 列块每步余量（< tPRB）
    const uint64_t rowStep = stride / tprb;  // 行带每步基数
    // 本 block 首 tile 的二维坐标（每 kernel 一次除法；各 batch 的领取序列
    // 相同，仅 batch 维不同，故每 batch 复用该初值）。
    const uint64_t rowBand0 = blockIdx / tprb;
    const uint64_t colChunk0 = blockIdx % tprb;
    for (uint64_t batch = 0; batch < tilingData_.numBatches; ++batch) {
        uint64_t tim = blockIdx;
        uint64_t rowBand = rowBand0;
        uint64_t colChunk = colChunk0;
        while (tim < tpm) {
            const TilePosition tile = MakeTilePosition(tilingData_, batch, rowBand, colChunk);
            CopyIn(tile);
            Compute(tile);
            // V 写出（StoreAlign acc）对标量 GetValue 可见：PipeBarrier 排空 V pipe，
            // 再以 V→S 事件同步保证跨执行单元的写→读可见性（实测 VF 内向量写 UB 后
            // 仅 PipeBarrier 存在时序竞态，标量读会读到旧值/槽残留）。
            AscendC::PipeBarrier<PIPE_V>();
            SynchronizePipeline<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < lanes; ++i) {
                blockViolation = blockViolation || (accLocal_.GetValue(i) != 0);
            }
            // 下一 tile 搬入前保护 inputBuf WAR（上一 tile 的 VF 读取先于覆写完成）。
            SynchronizePipeline<AscendC::HardEvent::V_MTE2>();
            // 增量推进（零除模）：colChunk 单次条件减法（colChunk + colStep < 2·tPRB
            // 恒成立）；rowBand 允许越界增长，但仅在 tim < tPM 时被读取，越界值
            // 不消费，且每 batch 重置回初值。
            tim += stride;
            colChunk += colStep;
            if (colChunk >= tprb) {
                colChunk -= tprb;
                ++rowBand;
            }
            rowBand += rowStep;
        }
    }
    // d_valid 多 block 归约：仅违规 block 写 1（写同值，无写竞争；非违规 block 不写）。
    if (blockViolation) {
        dValidGm_.SetValue(0, 1);
    }
}

template <typename U>
__aicore__ inline void PruneCheckKernel<U>::CopyIn(const TilePosition& tile)
{
    auto* input = reinterpret_cast<__ubuf__ U*>(inputLocal_.GetPhyAddr());
    const uint32_t pitchElems = tilingData_.inputPitch / sizeof(U);
    // O2 必落地：列尾块条件预清零 —— 仅 validCols < tileCols 的块执行 Clear 扫描；
    // 整块 DataCopyPad 完整覆盖 slot，跳过清零（消除内部整块的清零 V 开销）。
    if (tile.validCols < tilingData_.tileCols) {
        asc_vf_call<PruneCheckClearInput<U>>(input, tilingData_.tileRows * pitchElems);
        // 清零结果对 MTE2 可见后才可发起 DMA 覆写（spmma_compress CopyIn 同款）。
        SynchronizePipeline<AscendC::HardEvent::V_MTE2>();
    }
    // 多行单次搬运：blockLen 恒为 32B 倍数（维度对齐保证），dstStride 单位 32B，
    // 行距强制为 256B 对齐 pitch（prune SpltPruneLoadRowBlockChunk 凑 pitch 同型算法）。
    // 源地址 = d_in + (batch×batchStride + rowOffset×ld + colOffset)：batch 项为
    // 16B/dataBytes 元素倍数（host #6 保证，batch 对齐域 16B 与 prune 同口径）、
    // 行/列项为 32B/dataBytes 元素倍数（host #9 维度对齐保证）→ d_in 16B 对齐
    //（#8）时各 tile 源地址恒 16B 对齐（DataCopyPad 实测容忍 16B 对齐 GM 源：
    // D5 轮 d_in 16B 基址与逐 batch 16B 源探针全过，回归锚点 DinAlign16B 族
    // 与 MultiBatchStride16BGranularity* 批量用例）。
    const uint32_t blockLen = static_cast<uint32_t>(tile.validCols * sizeof(U));
    const uint32_t naturalBytes = (blockLen + 31) / 32 * 32;
    const AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(tile.validRows), blockLen,
        static_cast<int64_t>((tilingData_.ld - tile.validCols) * sizeof(U)),
        static_cast<int64_t>((tilingData_.inputPitch - naturalBytes) / 32), 0};
    const AscendC::DataCopyPadExtParams<U> padParams{true, 0, 0, static_cast<U>(0)};
    // batch 基址偏移 batchStride 元素（host #6 已保证 16B/dataBytes 对齐，
    // 各 batch 源地址保持 ≥ 16B 对齐——16B 非 32B 逐 batch 源已经 D5 轮
    // 实测，与 batch 0 的对齐性质在 DataCopyPad 实测容忍域内一致）；
    // numBatches==1 时偏移恒 0。
    AscendC::DataCopyPad(inputLocal_,
                inputGm_[tile.batch * tilingData_.batchStride + tile.rowOffset * tilingData_.ld +
                          tile.colOffset],
                copyParams, padParams);
    // 搬入完成对 V 可见后才做向量加载。
    SynchronizePipeline<AscendC::HardEvent::MTE2_V>();
}

template <typename U>
__aicore__ inline void PruneCheckKernel<U>::Compute(const TilePosition& tile)
{
    auto* input = reinterpret_cast<__ubuf__ U*>(inputLocal_.GetPhyAddr());
    auto* acc = reinterpret_cast<__ubuf__ U*>(accLocal_.GetPhyAddr());
    const uint32_t pitchElems = tilingData_.inputPitch / sizeof(U);
    if (tilingData_.alongRow) {
        asc_vf_call<PruneCheckRowGroups<U>>(input, acc, pitchElems, tile.validRows, tile.validCols);
    } else {
        asc_vf_call<PruneCheckColGroups<U>>(input, acc, pitchElems, tile.validRows, tile.validCols);
    }
}
} // namespace

// ============================================================================
// Kernel 入口（每 dtype 一个），host launcher 按 tiling.dataBytes 分派
// （spmma_compress kernel_do 同款分派）。tiling by-value 经 launch args 传递。
// ============================================================================
extern "C" __global__ __aicore__ void prune_check_u32(GM_ADDR input, GM_ADDR dValid,
    PruneCheckTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    PruneCheckKernel<uint32_t> kernel;
    kernel.Init(input, dValid, tiling, pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void prune_check_u16(GM_ADDR input, GM_ADDR dValid,
    PruneCheckTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    PruneCheckKernel<uint16_t> kernel;
    kernel.Init(input, dValid, tiling, pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void prune_check_u8(GM_ADDR input, GM_ADDR dValid,
    PruneCheckTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    PruneCheckKernel<uint8_t> kernel;
    kernel.Init(input, dValid, tiling, pipe);
    kernel.Process();
}

extern "C" void splt_prune_check_kernel_do(GM_ADDR input, GM_ADDR dValid,
    uint32_t blockDim, const PruneCheckTilingData& tiling, void* stream)
{
    if (tiling.dataBytes == 4) {
        prune_check_u32<<<blockDim, nullptr, stream>>>(input, dValid, tiling);
    } else if (tiling.dataBytes == 2) {
        prune_check_u16<<<blockDim, nullptr, stream>>>(input, dValid, tiling);
    } else {
        prune_check_u8<<<blockDim, nullptr, stream>>>(input, dValid, tiling);
    }
}
