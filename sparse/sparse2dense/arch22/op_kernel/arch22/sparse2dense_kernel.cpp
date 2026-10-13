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


// ============================================================================
// sparse2dense_kernel.cpp — aclsparse_sparse2dense 主线 kernel
//   scope：TilingKey_0~4（全 5 dtype）双 kernel + Host 分发 + bufferSize 恒 0 协议
//
// 双 kernel：
//   清零 kernel sparse2dense_zero_kernel_<dtype>       —— 清零（逐 TilingKey 宏展开入口）：matB 全 buffer（含 ld padding）写
//       对应 dtype 正零 bit 模式；各核 GetBlockIdx/GetBlockNum 32B 粒度对齐分片，
//       UB 造零块（按 V_KIND 编译期选择通道）+ 对齐主体 DataCopy 大块写 + 尾块
//       DataCopyPad 字节粒度写。全 5 dtype 通道完整实现（全 dtype 每次执行必触发）。
//   scatter kernel sparse2dense_scatter_kernel_<dtype> —— scatter（逐 TilingKey 宏展开入口）：CSR/CSC 段块主轴（SEG_BLOCK_SEGS）
//       × 段内元素 tile（NNZ_TILE）两级分片 / COO 元素块一级分片；DataCopyPad 读入
//       offsets/indices/values 分片（MTE2，双缓冲），anchor=offsets[0] 相对差值展开
//       （offsets 不减 base）、坐标减 indexBase、ROW/COL 寻址、连续 run 识别与
//       UB 打包（Copy 对齐快速路径 / SetValue 兜底），每 run 一次 DataCopyPad 写 GM
//       （MTE3）+ PipeBarrier<PIPE_MTE3>。
//       转置组合（CSC×ROW / CSR×COL）三层写路径：
//       桶转置路径：scatter kernel Init 依「联合成本门」判定 bucketActive = 转置组合 × 非
//       complex64 × est ≤ 238ns（0.85×基线 280ns/元素；est = W_eff×S + H +
//       seq×F/elemsPerCore + M，S=33/H=70/F=7μs/M=20 性能锚点，Host tiling 与
//       scatter kernel Init 共用 sparse2dense_tiling_data.h 的 S2DBucketGateActive 同式纯元数据
//       判定）。激活实例改走
//       「列子块（C=A=32/elemSize，行段恒 32B）× 窗口批（rowTile=256、W_CO=16 窗共存，
//       窗口单元 8KB）」单遍多窗聚合路径——批内 16 窗 tile 清零（+0 与清零 kernel 幂等）+ 子块
//       元素单遍扫描 SetValue 聚合 + occ 首触哨兵去重（occList 紧凑追加）→ PipeBarrier
//       <PIPE_ALL> 后 occList 紧凑 flush（每占用行一次恒 32B 常长段写）+ PipeBarrier
//       <PIPE_MTE3>（每批一对屏障 + 窗口复用前导 PIPE_ALL）；W_eff>
//       1（rows>4096）按窗口批重扫，批数计入联合成本门。TPipe 运行时互斥分配
//       bucketTile(16×8KB=128KB)+occ/occList(16KB)，UB 复算 ≤184KB；W_CO 为编译期单点
//       降档护栏（降档 → 预算同比缩小、门激活域自动收窄）。未激活组合回退原 run 打包
//       路径，行为不变。
//       TK0~TK4 全 dtype 完整实现（搬运通道 traits 见 S2DChanTraits）。
//       批量散布直写快路径（仅承接桶门排除域 est>238ns，判定见 ProcessV1 注）：
//       段块主轴不变 + 块内列段分组列号摊销（列游标
//       沿 offBuf 每段解码一次列号）+ 32B 槽位 [value@w0, gmOff 64 位 lo/hi@w1–w2]
//       逐元素 SetValue 打包（packSlot 双缓冲 64KB，tileElems=1024、槽距恒 32B）+
//       每 tile 一对屏障（PIPE_ALL：Scalar 打包→MTE3 读；PIPE_MTE3：MTE3 完成→槽复用）
//       + 按槽序每元素一次单 burst DataCopyPad 直写 dense[(r×ld+c)×elemSize]
//       （blockLen=elemSize ≤ 8B 天然满足 blockLen≤65535B、GM 任意字节地址、UB 槽基址
//       32B 对齐、gmOff 64 位承接 rows·ld·elemSize 可达 2⁶²）；不引入合并/多 burst；
//       写域恰覆盖坐标位置、不触碰 ld padding，gap 由清零 kernel 承接。
//       CSR×COL 与 CSC×ROW 结构对偶（基线路径下 run 恒长度 1、逐元素双 PIPE_ALL 屏障
//       ~260ns/元素），桶/直写两层路径对称覆盖两组合——门条件泛化为 transposed、
//       窗口批维 winDim 按 layout 取 rows/cols、GM 寻址泛化为
//       (secIdx×ld+segIdx)×elemSize 单点表达式（批维/子块划分纯元数据、元素→桶映射
//       为存储序确定函数、跨核写域两两不交）。
//
// 平台约束（dav_c220/CANN 9.1.0）：
//   1) Copy<int8_t>（UB→UB mask 连续模式）对 8 位类型静默失效
//      ——8 位通道打包必须 SetValue 逐元素（float32 4B 通道 Copy 正常）；
//   2) 打包缓冲 Scalar 写 → MTE3 读的跨流水可见性需屏障承接
//      （Copy 路径 PipeBarrier<PIPE_V>，写出 PipeBarrier<PIPE_MTE3>）；
//   3) scatter kernel grid-stride 预取下一块前必须以局部参数锁定当前块元素数（CopyInBlock(next)
//      会覆写队列状态；仅 blockNum>核数且尾块非满时触发）——COO 主路径沿用此写法；
//      CSR/CSC 块内多 tile，跨块预取与内层 tile 流水冲突，结构上按块同步处理规避；
//   4) DataCopyExtParams.blockLen 单块上限按 65535B 保守约束（tileElems=1024 → 4KB）；
//   5) 核数动态获取（ACL_DEV_ATTR_VECTOR_CORE_NUM，不硬编码）；
//   6) host 侧 nnz=0 空 vector memcmp/上传须守卫。
//
// 禁用废弃宏 TILING_KEY_IS / BEGIN_TILING_DATA_DEF。
// ============================================================================

// sparse2dense_tiling_data.h 须先于 kernel_operator.h 包含（并入 TilingKey 声明后其
// template_argument.h 与 kernel_operator.h 的 AscendC::printf 同 TU 相遇，先含 host API
// 侧可使其函数体在 AscendC::printf 声明前解析；host.cpp 同序）
#include "sparse2dense_tiling_data.h"
#include "kernel_operator.h"
#include "sparse2dense_kernel.h"
#include "sparse2dense_kernel_entries.h"  // 内核入口声明（与文件尾宏展开定义单点对齐）

using namespace AscendC;

namespace {

// Copy（UB→UB run 打包）单次 mask 上限
constexpr uint32_t S2D_COPY_MASK_MAX = 64;

// 将字节数向下取整到 32B 倍数（device 侧可调用，需 __aicore__ 标注）
constexpr __aicore__ inline uint64_t FloorAlign32(uint64_t v)
{
    return v & ~static_cast<uint64_t>(31);
}

}  // namespace

// ===========================================================================
// 清零 kernel（位宽参数化，全 5 dtype 通道完整实现）
// ===========================================================================
template <uint32_t V_KIND>
class KernelS2DZero {
public:
    __aicore__ inline KernelS2DZero() {}

    __aicore__ inline void Init(GM_ADDR denseGmAddr, const Sparse2DenseTilingData &tiling)
    {
        totalBytes_ = static_cast<uint64_t>(tiling.zeroTotalBytes);
        tileBytes_ = tiling.zeroTileBytes;

        // UB 预算（清零 kernel 零块双缓冲 16KB、scatter kernel 全缓冲 ≈56KB、
        // complex64 8B 通道 ≈72KB，均 ≤ 192KB，余量 >2.5×）；zeroTileBytes 由 Host 按基准
        // 下发，device 侧保持 32B 下限。
        // （不做 GetCoreMemSize 运行时收窄：CANN 9.1.0 平台
        // 访问器 GetInstance 为 host-only（std::mutex），__aicore__ 上下文不可调用；
        // 设计期预算对全部合法输入有确定上界。）
        if (tileBytes_ < 32U) {
            tileBytes_ = 32U;  // rows/cols >= 1 ⇒ 至少一个 32B 块
        }

        denseGm_.SetGlobalBuffer((__gm__ int8_t *)denseGmAddr, totalBytes_);
        pipe_.InitBuffer(zeroQue_, SPARSE2DENSE_BUFFER_NUM, tileBytes_);

        // 各核边界按 32B 粒度向上取整对齐（不足 32B 余量并入尾侧），保证每核 begin 32B 对齐；
        // 区间不足以整分时部分核区间为空
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
        uint64_t perCore = (totalBytes_ + coreNum - 1UL) / coreNum;
        perCore = (perCore + 31UL) / 32UL * 32UL;
        begin_ = static_cast<uint64_t>(blockIdx) * perCore;
        end_ = begin_ + perCore;
        if (end_ > totalBytes_) {
            end_ = totalBytes_;
        }
    }

    __aicore__ inline void Process()
    {
        if (begin_ >= end_) {
            return;  // 区间为空（部分核快速返回）
        }
        for (uint64_t cur = begin_; cur < end_;) {
            uint64_t len = end_ - cur;
            if (len > static_cast<uint64_t>(tileBytes_)) {
                len = tileBytes_;
            }
            ZeroTile(cur, len);
            cur += len;
        }
    }

private:
    // 造零通道按 V_KIND 编译期选择：
    //   int8 → uint16（Duplicate 不含 int8，0x0000 ≡ 2×int8 0x00）；float16 → half；
    //   bfloat16 → bfloat16_t；float32 → float；complex64 → uint32（2×float32，+0f 全零）
    __aicore__ inline void FillZeroChannel(LocalTensor<int8_t> z8, uint32_t channelUnits)
    {
        if constexpr (V_KIND == SPARSE2DENSE_VK_INT8) {
            LocalTensor<uint16_t> z = z8.ReinterpretCast<uint16_t>();
            Duplicate<uint16_t>(z, static_cast<uint16_t>(0U), channelUnits);
        } else if constexpr (V_KIND == SPARSE2DENSE_VK_FLOAT16) {
            LocalTensor<half> z = z8.ReinterpretCast<half>();
            Duplicate<half>(z, static_cast<half>(0.0f), channelUnits);
        } else if constexpr (V_KIND == SPARSE2DENSE_VK_BFLOAT16) {
            LocalTensor<bfloat16_t> z = z8.ReinterpretCast<bfloat16_t>();
            Duplicate<bfloat16_t>(z, static_cast<bfloat16_t>(0.0f), channelUnits);
        } else if constexpr (V_KIND == SPARSE2DENSE_VK_FLOAT32) {
            LocalTensor<float> z = z8.ReinterpretCast<float>();
            Duplicate<float>(z, 0.0f, channelUnits);
        } else {  // SPARSE2DENSE_VK_COMPLEX64：uint32 通道，实/虚两分量均为 +0f（bit 全零）
            LocalTensor<uint32_t> z = z8.ReinterpretCast<uint32_t>();
            Duplicate<uint32_t>(z, 0U, channelUnits);
        }
    }

    // 单零块写出：对齐主体 DataCopy 大块写（32B 对齐快速路径）+ 尾块 DataCopyPad
    // 字节粒度写（奇字节尾块在此承接，如 rows=cols=ld=1）
    __aicore__ inline void ZeroTile(uint64_t gmOff, uint64_t bytes)
    {
        // 造零通道单元字节数（编译期常量，V_KIND 为模板参数）：
        //   int8 → uint16（2B，覆盖 2×int8）；fp16/bf16 → 2B 通道；fp32/complex64 → 4B 通道
        constexpr uint32_t kChannelBytes = static_cast<uint32_t>(
            (V_KIND == SPARSE2DENSE_VK_INT8)
                ? 2U
                : ((V_KIND == SPARSE2DENSE_VK_FLOAT16 || V_KIND == SPARSE2DENSE_VK_BFLOAT16) ? 2U : 4U));
        LocalTensor<int8_t> zero = zeroQue_.AllocTensor<int8_t>();
        uint32_t channelUnits = static_cast<uint32_t>(tileBytes_ / kChannelBytes);
        FillZeroChannel(zero, channelUnits);
        zeroQue_.EnQue<int8_t>(zero);
        LocalTensor<int8_t> zeroDeq = zeroQue_.DeQue<int8_t>();

        uint64_t aligned = FloorAlign32(bytes);
        if (aligned > 0UL) {
            DataCopy(denseGm_[gmOff], zeroDeq, static_cast<uint32_t>(aligned));
        }
        uint64_t tail = bytes - aligned;
        if (tail > 0UL) {
            DataCopyPad(denseGm_[gmOff + aligned], zeroDeq, {1, static_cast<uint32_t>(tail), 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE3>();
        zeroQue_.FreeTensor(zeroDeq);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECOUT, SPARSE2DENSE_BUFFER_NUM> zeroQue_;
    GlobalTensor<int8_t> denseGm_;
    uint64_t totalBytes_;
    uint64_t begin_;
    uint64_t end_;
    uint32_t tileBytes_;
};

// ===========================================================================
// scatter kernel（CSR/CSC 段块 × 段内 tile 两级分片 / COO 元素块一级分片）
// ===========================================================================
// scatter kernel（CSR/CSC 段块 × 段内元素 tile 两级分片 / COO 元素块一级分片）
//   全 5 dtype（TK0~TK4）通道完整实现。搬运通道按
//   V_KIND 编译期选择（纯 bit 搬运，无算术无类型转换，NaN/±inf/±0 bit 保持）：
//     TK0 int8      → int8_t  通道 ×1 单元/元素（Copy<int8_t> 在 dav_c220/CANN 9.1.0
//                     对 8 位类型静默失效——run 打包
//                     SetValue 逐单元 + PIPE_ALL）
//     TK1/TK2 fp16/bf16 → uint16_t 通道 ×1 单元/元素（SetValue 标量安全路径，同 8 位
//                     通道口径 PIPE_ALL）
//     TK3 float32   → uint32_t 通道 ×1 单元/元素（Copy 快速
//                     路径 PIPE_V + SetValue 兜底）
//     TK4 complex64 → uint32_t 通道 ×2 单元/元素（GM 偏移 ×2、
//                     Copy 快速路径 + SetValue 兜底）
// ===========================================================================

// scatter kernel 搬运通道 traits（编译期常量；kUnitsPerElem：每元素的通道单元数；
// kCopyFast：Copy（UB→UB）快速路径是否可用——仅 4B 通道启用）
template <uint32_t K> struct S2DChanTraits;
template <> struct S2DChanTraits<SPARSE2DENSE_VK_INT8> {
    using T = int8_t;
    static constexpr uint32_t kUnitsPerElem = 1U;
    static constexpr bool kCopyFast = false;  // Copy<int8_t> 对 8 位类型静默失效
};
template <> struct S2DChanTraits<SPARSE2DENSE_VK_FLOAT16> {
    using T = uint16_t;
    static constexpr uint32_t kUnitsPerElem = 1U;
    static constexpr bool kCopyFast = false;  // 2B 通道走 SetValue 标量安全路径
};
template <> struct S2DChanTraits<SPARSE2DENSE_VK_BFLOAT16> {
    using T = uint16_t;
    static constexpr uint32_t kUnitsPerElem = 1U;
    static constexpr bool kCopyFast = false;  // 2B 通道走 SetValue 标量安全路径
};
template <> struct S2DChanTraits<SPARSE2DENSE_VK_FLOAT32> {
    using T = uint32_t;
    static constexpr uint32_t kUnitsPerElem = 1U;
    static constexpr bool kCopyFast = true;   // 4B 通道 Copy 可用
};
template <> struct S2DChanTraits<SPARSE2DENSE_VK_COMPLEX64> {
    using T = uint32_t;
    static constexpr uint32_t kUnitsPerElem = 2U;  // 8B = 2×uint32
    static constexpr bool kCopyFast = true;
};

template <uint32_t V_KIND>
class KernelS2DScatter {
public:
    using Chan = typename S2DChanTraits<V_KIND>::T;
    static constexpr uint32_t kUnitsPerElem = S2DChanTraits<V_KIND>::kUnitsPerElem;
    static constexpr bool kCopyFast = S2DChanTraits<V_KIND>::kCopyFast;
    // Copy 快速路径 32B 对齐单元数（4B 通道 = 8 单元）
    static constexpr uint32_t kCopyAlignUnits = 32U / static_cast<uint32_t>(sizeof(Chan));

    // ---- 桶转置路径编译期参数（complex64 编译期排除）----
    //   rowTile=256、W_CO=16 窗共存（窗口单元 = rowTile×C×elemSize = 256×32B = 8KB）；
    //   C=A=32/elemSize（fp32 8 / fp16·bf16 16 / int8 32）→ 窗口行段恒 32B（对齐、常长）；
    //   批 = W_CO 窗 × rowTile 行 = 4096 行（2 的幂 → 行→（批,窗口,行内）映射全位移）；
    //   桶 tile 总量 = W_CO×8KB = 128KB，与 occ 哨兵/occList（2×8KB）合计 144KB ≤ 184KB。
    //   W_CO 降档护栏：单点参数，降档 → bucketTile 预算同比缩小、S2DBucketWinBatches 的
    //   W_eff 同比增大 → 联合成本门激活域自动收窄（设计期预算口径，不做运行时收窄）。
    static constexpr bool kBucketSupported = (V_KIND != SPARSE2DENSE_VK_COMPLEX64);
    static constexpr uint32_t kElemBytes =
        static_cast<uint32_t>(sizeof(Chan)) * kUnitsPerElem;           // 元素字宽 1/2/4/8B
    static constexpr uint32_t kRowTileR2 = 256U;                      // 行窗口高（2 的幂）
    static constexpr uint32_t kRowTileLogR2 = 8U;                     // log2(rowTile)
    static constexpr uint32_t kWinCo = 16U;                           // 共存窗口数 W_CO
    static constexpr uint32_t kAlignElems = 32U / kElemBytes;         // A = 32/elemSize
    static constexpr uint32_t kColChunk = kAlignElems;                // C = A：列子块宽（恒 32B 行段）
    static constexpr uint32_t kWinTileElems = kRowTileR2 * kColChunk; // 窗口单元元素数
    static constexpr uint32_t kWinTileBytes = kWinTileElems * kElemBytes;  // = 8192（各 dtype 一致）
    static constexpr uint32_t kBatchRows = kRowTileR2 * kWinCo;       // 单批覆盖行数 4096（2 的幂）
    static constexpr uint32_t kBatchLog = kRowTileLogR2 + 4U;         // r>>kBatchLog == 批号（12）
    // 满子块行段恒 32B（=kColChunk×kElemBytes）；尾子块按 cSpan×elemSize 收窄（flush 见注）

    // ---- 批量散布直写层编译期参数----
    //   槽距恒 32B（全 dtype 统一，槽内 [value@w0, gmOff 64 位 lo/hi@w1–w2, 预留@w3]）；
    //   packSlot 双缓冲 = tileElems×32B×2 = 64KB（tileElems=1024）；tileElems 降档护栏
    //   （可用 <184KB 时降 512 → packSlot 32KB、屏障对翻倍）由 Host 下发
    //   tileElems 字段单点承接（packSlot 容量按 tileElems_ 同比缩放），UB 192KB
    //   > 184KB 保守线维持 1024。
    static constexpr uint32_t kSlotBytes = 32U;                       // 直写快路径槽距（B）
    static constexpr uint32_t kSlotUnits =
        kSlotBytes / static_cast<uint32_t>(sizeof(Chan));             // 槽距的 Chan 单元数（w0 位置）
    // gmOff 64 位 lo 的槽内 int32 单元号：4B 及以下值占 u0 → 偏移@u1–u2；
    // complex64 8B 值占 u0–u1 → 偏移@u2–u3（槽距恒 32B 不变）
    static constexpr uint32_t kSlotOffUnit = kUnitsPerElem;

    __aicore__ inline KernelS2DScatter() {}

    __aicore__ inline void Init(GM_ADDR valGmAddr, GM_ADDR primGmAddr, GM_ADDR secGmAddr,
        GM_ADDR denseGmAddr, const Sparse2DenseTilingData &tiling)
    {
        rows_ = tiling.rows;
        cols_ = tiling.cols;
        ld_ = tiling.ld;
        nnz_ = tiling.nnz;
        lenPrimary_ = tiling.lenPrimary;
        format_ = tiling.format;
        indexBase_ = static_cast<int64_t>(tiling.indexBase);
        isColMajor_ = tiling.isColMajor;
        segNum_ = tiling.segNum;
        segsPerBlock_ = tiling.segsPerBlock;
        tileElems_ = tiling.tileElems;
        blockNum_ = tiling.blockNum;

        valGm_.SetGlobalBuffer((__gm__ Chan *)valGmAddr, static_cast<uint64_t>(nnz_) * kUnitsPerElem);
        primGm_.SetGlobalBuffer((__gm__ int32_t *)primGmAddr, lenPrimary_);
        secGm_.SetGlobalBuffer((__gm__ int32_t *)secGmAddr, nnz_);
        offGm_.SetGlobalBuffer((__gm__ int32_t *)primGmAddr, lenPrimary_);  // CSR/CSC offsets 读（同 primary 地址）
        denseGm_.SetGlobalBuffer((__gm__ Chan *)denseGmAddr);
        // 直写快路径 dense 字节视图（gmOff 64 位字节偏移；GM 任意字节地址由 DataCopyPad 承接）
        denseByteGm_.SetGlobalBuffer((__gm__ int8_t *)denseGmAddr);

        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(GetBlockNum());

        // 桶层联合成本门（转置组合对称覆盖）：CSC×ROW 或
        // CSR×COL（transposed，两组合 secondary/段维角色互换、GM 寻址同为
        // (sec×ld+seg)×elemSize）× 非 complex64 × est ≤ 238ns（0.85×基线 280ns/元素）。
        // est = W_eff×S + H + seq×F/elemsPerCore + M（S=33/H=70/F=7μs/M=20 性能锚点），
        // 经 sparse2dense_tiling_data.h 的 S2DBucketGateActive 与 Host tiling 同式纯元数据
        // 判定（coreNum=GetBlockNum() 即 launch grid=min(blockNum,
        // 核数)）。窗口批维 winDim：ROW 布局取 rows（CSC×ROW）、COL 布局取 cols（CSR×COL）。
        // 激活条件只依赖 tiling 元数据与核数、不依赖数据内容 → 同输入同路径同划分（确定性
        // 前提）；激活域外（CSR×ROW/CSC×COL/COO/门超预算/complex64）回退原 run 打包路径。
        // 转置组合判定与寻址泛化说明：CSR×COL 与 CSC×ROW 结构对偶（scatter kernel 基线路径下两者
        // run 恒长度 1、逐元素两配 PIPE_ALL 屏障 ~260ns/元素），扩展后
        // 桶层批维=secondary（CSC=行/CSR=列）、子块维=段（CSC=列/CSR=行）。
        transposed_ = (format_ == SPARSE2DENSE_FMT_CSC && isColMajor_ == 0U) ||
            (format_ == SPARSE2DENSE_FMT_CSR && isColMajor_ != 0U);
        winDim_ = (isColMajor_ == 0U) ? rows_ : cols_;
        bucketActive_ = kBucketSupported && transposed_ && S2DBucketGateActive(
            true, true, static_cast<uint64_t>(winDim_), nnz_, blockNum_,
            segsPerBlock_, kColChunk, kRowTileR2, kWinCo, coreNum_);
        winBatches_ = bucketActive_ ? static_cast<uint32_t>(
            S2D_GATE_WIN_BATCHES(static_cast<uint64_t>(winDim_), kRowTileR2, kWinCo)) : 0U;

        // 直写快路径层激活判定（转置组合三层判定第三支；complex64 转置组合纳入直写快路径——8B 值
        // 通道槽位扩展 [valLo@u0, valHi@u1, offLo@u2,
        // offHi@u3]，槽距恒 32B 不变）：CSC×ROW / CSR×COL 且桶门未激活 → 直写快路径（非
        // complex64 为桶门排除域 est>238ns 或门退化；complex64 由 kBucketSupported=false
        // 恒排除桶层）。
        // 纯元数据判定（format/isColMajor/valueKind/rows/nnz/blockNum + 核数），与 Host
        // FillTiling 同式可复算（门公式单点承载于 sparse2dense_tiling_data.h），不新增
        // TilingData 字段；nnz=0 先于一切路径判定在 Process() 返回。桶转置路径（bucketActive_）
        // 与直写快路径（v1Active_）互斥完备覆盖转置组合全 dtype 域（同输入唯一路径，确定性前提）。
        v1Active_ = transposed_ && !bucketActive_;

        // 锚定点：anchor = offsets[0]，段定位/展开以相对差值承接
        //（禁止用 raw offsets 与元素下标直接比较，base=1 时整体 off-by-one）
        if (format_ != SPARSE2DENSE_FMT_COO) {
            anchor_ = primGm_.GetValue(0);
        }

        InitUb();
    }

    __aicore__ inline void Process()
    {
        if (nnz_ <= 0 || blockNum_ == 0) {
            return;  // NNZ=0：scatter kernel 直接返回，输出全定义由清零 kernel 承接
        }
        if (format_ == SPARSE2DENSE_FMT_COO) {
            ProcessCoo();
        } else if (bucketActive_) {
            ProcessBucket();  // CSC×ROW 桶转置打包路径（先于路径判定的 NNZ=0 返回承接）
        } else if (v1Active_) {
            ProcessV1();  // 转置组合桶门排除域 → 批量散布直写快路径
        } else {
            ProcessSeg();  // 基线路径：CSR/COO run 合并、CSC×COL colreg、complex64 逐元素
        }
    }

private:
    // ---- UB 缓冲（通道宽度参数化，双缓冲 BUFFER_NUM=2）----
    // 容量按设计期预算口径（不运行时 GetCoreMemSize 收窄，见清零 kernel Init 注释）：
    // scatter kernel 全缓冲 = values/pack（tileElems×单元宽×2）+ secondary/COO primary（8KB）+
    // CSR/CSC offBuf（≤72B×... 上界确定），complex64 8B 通道最坏 ≈72KB ≤ UB 192KB。
    __aicore__ inline void InitUb()
    {
        // values/打包分片：tileElems × kUnitsPerElem × sizeof(Chan)（int8:1B / 2B 通道:2B /
        // float32:4B / complex64:8B）；secondary 分片 8KB
        pipe_.InitBuffer(valQue_, SPARSE2DENSE_BUFFER_NUM, tileElems_ * kUnitsPerElem * sizeof(Chan));
        pipe_.InitBuffer(secQue_, SPARSE2DENSE_BUFFER_NUM, tileElems_ * sizeof(int32_t));
        if (format_ == SPARSE2DENSE_FMT_COO) {
            // COO：primary 逐点分片随块读入（8KB）
            pipe_.InitBuffer(primQue_, SPARSE2DENSE_BUFFER_NUM, tileElems_ * sizeof(int32_t));
        } else {
            // CSR/CSC：块 offsets 区间一次性读入，项数上界 = segsPerBlock + 1
            //（确定上界，与 nnz/空段分布无关），块内所有内层 tile 复用
            uint32_t offCap = (segsPerBlock_ + 1U + 7U) / 8U * 8U;
            pipe_.InitBuffer(offQue_, 1, offCap * sizeof(int32_t));
        }
        if (bucketActive_) {
            // 单遍多窗桶实例：运行时不分配 packBuf，改配 bucketTile（W_CO×窗口单元 8KB
            // = 128KB）+ occ 首触哨兵/occList（2×8KB=16KB）——与基线实例按 tiling 分支互斥
            // 分配。UB 复算（fp32 最坏）：128 + 16 + val/sec 双缓冲 16KB + offBuf ≤184KB
            // 保守线（W_CO 降档护栏见编译期参数注）。
            if constexpr (kBucketSupported) {
                pipe_.InitBuffer(bucketTbuf_, kWinCo * kWinTileBytes);
                pipe_.InitBuffer(occTbuf_, kBatchRows * sizeof(uint16_t));
                pipe_.InitBuffer(listTbuf_, kBatchRows * sizeof(uint16_t));
            }
        } else if (v1Active_) {
            // 直写快路径实例：运行时不分配基线 packBuf，改配 32B 槽位打包 packSlot
            // （tileElems×32B×双缓冲 = 64KB，槽距恒 32B 全 dtype 统一）。UB 复算 fp32
            // ≈88.5KB ≤ 184KB 保守线（values 8 + secondary 8 + offBuf 0.52 + packSlot 64
            // + 系统余量 ~8）；tileElems 降档护栏（极端情形降 512 → packSlot 32KB、
            // 合计 ≈56.5KB）由 Host tileElems 字段单点承接，容量在此同比缩放。
            pipe_.InitBuffer(packSlotQue_, SPARSE2DENSE_BUFFER_NUM,
                static_cast<uint32_t>(tileElems_) * kSlotBytes);
        } else {
            // 基线实例：打包缓冲 packBuf 双缓冲（run 合并后一次 DataCopyPad）
            pipe_.InitBuffer(packQue_, SPARSE2DENSE_BUFFER_NUM, tileElems_ * kUnitsPerElem * sizeof(Chan));
        }
    }

    // ---- 块/tile 元素数 ----
    // COO：blockNum = ceil(nnz/tileElems)，块 b 覆盖 [b*tileElems, min((b+1)*tileElems, nnz))
    __aicore__ inline uint32_t CooBlockElemCount(uint32_t b) const
    {
        uint64_t eBegin = static_cast<uint64_t>(b) * tileElems_;
        uint64_t eEnd = eBegin + tileElems_;
        if (eEnd > static_cast<uint64_t>(nnz_)) {
            eEnd = static_cast<uint64_t>(nnz_);
        }
        return static_cast<uint32_t>(eEnd - eBegin);
    }

    // ---- COO 路径（元素块一级分片 + grid-stride 预取式双缓冲流水）----
    // 预取下一块前必须以局部参数锁定当前块
    // 元素数（CopyInBlock(next) 会覆写队列状态；仅 blockNum>核数且尾块非满时触发）。
    __aicore__ inline void ProcessCoo()
    {
        if (blockIdx_ >= blockNum_) {
            return;
        }
        CopyInCooBlock(blockIdx_);  // 预取本核首块
        for (uint32_t b = blockIdx_; b < blockNum_; b += coreNum_) {
            uint32_t curN = CooBlockElemCount(b);  // ← 预取前锁定当前块元素数（CopyInBlock(next) 会覆写队列状态）
            valBuf_ = valQue_.DeQue<Chan>();
            primBuf_ = primQue_.DeQue<int32_t>();
            secBuf_ = secQue_.DeQue<int32_t>();
            uint32_t next = b + coreNum_;
            if (next < blockNum_) {
                CopyInCooBlock(next);  // 预取下一块，与当前块计算重叠
            }
            ComputeAndWrite(0, static_cast<int64_t>(curN));
            valQue_.FreeTensor(valBuf_);
            primQue_.FreeTensor(primBuf_);
            secQue_.FreeTensor(secBuf_);
        }
    }

    __aicore__ inline void CopyInCooBlock(uint32_t b)
    {
        uint64_t eBegin = static_cast<uint64_t>(b) * tileElems_;
        uint32_t n = CooBlockElemCount(b);
        uint32_t valBytes = n * kUnitsPerElem * sizeof(Chan);
        uint32_t idxBytes = n * sizeof(int32_t);

        LocalTensor<Chan> val = valQue_.AllocTensor<Chan>();
        DataCopyPad(val, valGm_[eBegin * kUnitsPerElem], {1, valBytes, 0, 0, 0}, {false, 0, 0, 0});
        valQue_.EnQue<Chan>(val);

        LocalTensor<int32_t> prim = primQue_.AllocTensor<int32_t>();
        DataCopyPad(prim, primGm_[eBegin], {1, idxBytes, 0, 0, 0}, {false, 0, 0, 0});
        primQue_.EnQue<int32_t>(prim);

        LocalTensor<int32_t> sec = secQue_.AllocTensor<int32_t>();
        DataCopyPad(sec, secGm_[eBegin], {1, idxBytes, 0, 0, 0}, {false, 0, 0, 0});
        secQue_.EnQue<int32_t>(sec);
    }

    // ---- CSR/CSC 路径（段块主轴 × 段内元素 tile 两级分片，按块同步处理）----
    __aicore__ inline void ProcessSeg()
    {
        for (uint32_t blk = blockIdx_; blk < blockNum_; blk += coreNum_) {
            ProcessSegBlock(blk);
        }
    }

    // ---- 段块区间定位（基线/桶转置/直写快路径三路径共用）：块 b 覆盖段
    // [b*segsPerBlock, min((b+1)*segsPerBlock, segNum))；块元素区间 [eBegin, eEnd) 由
    // 块端点 offsets 差值（anchor 锚定）定位。返回 false = 空块/越界块（整块全空段
    // 快速跳过，不产生 run）----
    __aicore__ inline bool SegBlockRange(uint32_t blk, int64_t &s0, int64_t &s1,
        int64_t &eBegin, int64_t &eEnd) const
    {
        s0 = static_cast<int64_t>(blk) * static_cast<int64_t>(segsPerBlock_);
        s1 = s0 + static_cast<int64_t>(segsPerBlock_);
        if (s1 > segNum_) {
            s1 = segNum_;
        }
        if (s0 >= s1) {
            return false;
        }
        // 块端点 offsets 4B GM 标量读；offsets 为长度语义，
        // 不参与 indexBase 换算；anchor = offsets[0] 锚定相对差值（base=1 天然成立）
        const int32_t o0 = offGm_.GetValue(s0);
        const int32_t o1 = offGm_.GetValue(s1);
        eBegin = static_cast<int64_t>(o0 - anchor_);
        eEnd = static_cast<int64_t>(o1 - anchor_);
        return eEnd > eBegin;
    }

    // ---- 块 offsets 区间 [s0, s1] 一次性读入 offBuf（≤ segsPerBlock+1 项，确定上界；
    // 三路径共用，块内全部内层 tile/子块/窗口批复用）----
    __aicore__ inline void LoadBlockOffsets(int64_t s0, int64_t s1)
    {
        const uint32_t offCount = static_cast<uint32_t>(s1 - s0 + 1);
        const uint32_t offBytes = offCount * sizeof(int32_t);
        LocalTensor<int32_t> offTmp = offQue_.AllocTensor<int32_t>();
        DataCopyPad(offTmp, offGm_[s0], {1, offBytes, 0, 0, 0}, {false, 0, 0, 0});
        offQue_.EnQue<int32_t>(offTmp);
        offLocal_ = offQue_.DeQue<int32_t>();
        segBase_ = static_cast<uint32_t>(s0);
        offCount_ = offCount;
    }

    __aicore__ inline void ProcessSegBlock(uint32_t blk)
    {
        int64_t s0 = 0;
        int64_t s1 = 0;
        int64_t eBegin = 0;
        int64_t eEnd = 0;
        if (!SegBlockRange(blk, s0, s1, eBegin, eEnd)) {
            return;  // 空块（整块全空段）快速跳过，不产生 run
        }

        LoadBlockOffsets(s0, s1);
        segCursor_ = 0;  // 段扫描游标跨内层 tile 保持单调推进

        // 段内元素 tile（二级轴）：单段 nnz 可远超 tileElems（极端：单行全密集 nnz=cols），
        // 由该段所在块的内层 tile 循环承载；run 不跨内层 tile（tile 边界自然闭合）
        int64_t span = eEnd - eBegin;
        int64_t innerTiles = (span + static_cast<int64_t>(tileElems_) - 1) / static_cast<int64_t>(tileElems_);
        for (int64_t t = 0; t < innerTiles; t++) {
            int64_t tBegin = eBegin + t * static_cast<int64_t>(tileElems_);
            int64_t tEnd = tBegin + static_cast<int64_t>(tileElems_);
            if (tEnd > eEnd) {
                tEnd = eEnd;
            }
            ProcessSegTile(tBegin, tEnd);
        }
        offQue_.FreeTensor(offLocal_);
    }

    __aicore__ inline void ProcessSegTile(int64_t tBegin, int64_t tEnd)
    {
        uint32_t n = static_cast<uint32_t>(tEnd - tBegin);
        tileBegin_ = tBegin;

        CopyInTile(tBegin, n);

        valBuf_ = valQue_.DeQue<Chan>();
        secBuf_ = secQue_.DeQue<int32_t>();

        ComputeAndWrite(tBegin, static_cast<int64_t>(n));

        valQue_.FreeTensor(valBuf_);
        secQue_.FreeTensor(secBuf_);
    }

    // 内层 tile 读入（MTE2，DataCopyPad 字节粒度；基线与桶路径共用）
    __aicore__ inline void CopyInTile(int64_t tBegin, uint32_t n)
    {
        uint32_t valBytes = n * kUnitsPerElem * sizeof(Chan);
        uint32_t idxBytes = n * sizeof(int32_t);

        LocalTensor<Chan> val = valQue_.AllocTensor<Chan>();
        DataCopyPad(val, valGm_[tBegin * kUnitsPerElem], {1, valBytes, 0, 0, 0}, {false, 0, 0, 0});
        valQue_.EnQue<Chan>(val);

        LocalTensor<int32_t> sec = secQue_.AllocTensor<int32_t>();
        DataCopyPad(sec, secGm_[tBegin], {1, idxBytes, 0, 0, 0}, {false, 0, 0, 0});
        secQue_.EnQue<int32_t>(sec);
    }

    // =========================================================================
    // 转置组合（CSC×ROW / CSR×COL）单遍多窗桶转置聚合路径
    //   （两组合 secondary/段维角色互换、
    //   GM 寻址同为 (批维×ld+段维)，批维 winDim：ROW 取 rows、COL 取 cols）
    //   主轴不变（段块 grid-stride 认领 + 块 offsets 区间一次读入复用）；块内按列子块
    //   （C=A=32/elemSize 列，窗口行段恒 32B）× 窗口批（rowTile=256、W_CO=16 窗共存）
    //   两级循环，对每个（子块 × 窗口批）：批内 16 窗 tile 清零（+0 与清零 kernel 幂等）→ 子块
    //   元素单遍扫描 SetValue 聚合 + occ 首触哨兵去重（occList 紧凑追加，u16 打包
    //   批内号 secIdx&4095）→ PipeBarrier<PIPE_ALL> → occList 紧凑 flush（每占用批维号一次
    //   恒 32B 常长段写，gap 为 +0 与清零 kernel 幂等）→ PipeBarrier<PIPE_MTE3>（每批一对
    //   屏障 + 窗口复用前导 PIPE_ALL）。W_eff>1（rows>4096）按窗口批
    //   重扫子块元素，批数 W_eff 与每核批数 seq 已计入联合成本门（Init 判定）。
    //   写序：激活/划分仅依赖 tiling 元数据与核数；元素→（子块,窗口,
    //   行内偏移）映射与 occList 追加序（扫描序）均为输入的确定函数；重复坐标合法
    //   （同核按扫描序先后 SetValue，结果取任一写入值）；跨子块/跨块列区间不相交 →
    //   跨核写区间两两不交；gap 覆写恒 +0（无重复坐标时同输入 bit-wise 一致）。
    // =========================================================================
    __aicore__ inline void ProcessBucket()
    {
        for (uint32_t blk = blockIdx_; blk < blockNum_; blk += coreNum_) {
            ProcessBucketBlock(blk);
        }
    }

    __aicore__ inline void ProcessBucketBlock(uint32_t blk)
    {
        // 主轴不变（段块区间定位/空块跳过见 SegBlockRange）
        int64_t s0 = 0;
        int64_t s1 = 0;
        int64_t eBegin = 0;
        int64_t eEnd = 0;
        if (!SegBlockRange(blk, s0, s1, eBegin, eEnd)) {
            return;
        }
        (void)eBegin;
        (void)eEnd;  // 桶路径仅用块端点判空，元素区间由子块差值展开

        // 块 offsets 区间 [s0, s1] 一次性读入 offBuf（≤ segsPerBlock+1 项），
        // 块内子块/窗口批全部复用
        LoadBlockOffsets(s0, s1);

        // 列子块（C=A 列，行段恒 32B）× 窗口批（W_CO=16 窗共存）两级循环；
        // 子块元素区间 [scLo,scHi) 由 offBuf 差值定位（无额外 GM offsets 读）
        const int64_t subChunks =
            (static_cast<int64_t>(s1) - s0 + static_cast<int64_t>(kColChunk) - 1) /
            static_cast<int64_t>(kColChunk);
        for (int64_t sc = 0; sc < subChunks; sc++) {
            const int64_t cLo = s0 + sc * static_cast<int64_t>(kColChunk);
            int64_t cHi = cLo + static_cast<int64_t>(kColChunk);
            if (cHi > s1) {
                cHi = s1;
            }
            const int64_t scLo =
                static_cast<int64_t>(offLocal_.GetValue(static_cast<uint32_t>(cLo - s0))) - anchor_;
            const int64_t scHi =
                static_cast<int64_t>(offLocal_.GetValue(static_cast<uint32_t>(cHi - s0))) - anchor_;
            if (scHi <= scLo) {
                continue;  // 空子块跳过
            }
            for (uint32_t b = 0; b < winBatches_; b++) {
                ProcessBucketBatch(cLo, cHi, scLo, scHi, b);
            }
        }
        offQue_.FreeTensor(offLocal_);
    }

    // 单（列子块 × 窗口批）处理：批内 16 窗 tile 清零 → 子块元素单遍扫描聚合（occ 首触
    // 哨兵去重）→ occList 紧凑 flush（恒 32B 常长段写）。屏障结构：窗口/occ 复用前导
    // PIPE_ALL（V 清零 vs 上一批 MTE3 段写
    // 读跨流水）→ V 清零 → PIPE_ALL（V→Scalar 写序）→ 扫描 → PIPE_ALL（Scalar 写→MTE3
    // 段写读）→ flush → PIPE_MTE3（段写完成），即每批一对 PIPE_ALL/PIPE_MTE3 + 前导屏障。
    __aicore__ inline void ProcessBucketBatch(int64_t cLo, int64_t cHi, int64_t scLo,
        int64_t scHi, uint32_t b)
    {
        const int64_t wBatch0 = static_cast<int64_t>(b) << kBatchLog;  // 批基 = b×4096（CSC×ROW 为行基 / CSR×COL 为列基）
        if (wBatch0 >= winDim_) {
            return;  // 尾批批维为空（winBatches_ 已按 winDim 上取整，保守守卫）
        }
        LocalTensor<Chan> tile = bucketTbuf_.Get<Chan>();
        LocalTensor<uint16_t> occ = occTbuf_.Get<uint16_t>();
        LocalTensor<uint16_t> occList = listTbuf_.Get<uint16_t>();

        PipeBarrier<PIPE_ALL>();  // 窗口/occ 复用前导（V 清零与上一批 MTE3 段写跨流水互斥）
        BucketClearBatch(tile, occ);
        PipeBarrier<PIPE_ALL>();  // V 清零 → Scalar SetValue 写序（跨 pipe 保守屏障）
        occCnt_ = 0;
        BucketScanFillBatch(tile, occ, occList, cLo, cHi, scLo, scHi, b);
        // Scalar 写 → MTE3 段写：PIPE_ALL 屏障
        PipeBarrier<PIPE_ALL>();
        BucketFlushOccList(tile, occList, cLo, cHi, wBatch0);
        PipeBarrier<PIPE_MTE3>();  // 段写完成
    }

    // 批清零：W_CO=16 窗口 tile（16×8KB=128KB）+ occ 首触哨兵（u16×4096）。+0 bit 模式
    // 与清零 kernel 造零同源（1B/2B 通道经 uint16 视角，4B 通道 uint32 视角）；逐窗口 8KB 清零，
    // 避免单次超大 repeat 次数（窗口单元 8KB = 各 dtype 一致）
    __aicore__ inline void BucketClearBatch(LocalTensor<Chan> &tile, LocalTensor<uint16_t> &occ)
    {
        if constexpr (sizeof(Chan) <= 2U) {
            LocalTensor<uint16_t> z = tile.template ReinterpretCast<uint16_t>();
            for (uint32_t w = 0; w < kWinCo; w++) {
                Duplicate<uint16_t>(z[w * (kWinTileBytes / 2U)], 0U, kWinTileBytes / 2U);
            }
        } else {
            LocalTensor<uint32_t> z = tile.template ReinterpretCast<uint32_t>();
            for (uint32_t w = 0; w < kWinCo; w++) {
                Duplicate<uint32_t>(z[w * (kWinTileBytes / 4U)], 0U, kWinTileBytes / 4U);
            }
        }
        Duplicate<uint16_t>(occ, 0U, kBatchRows);
    }

    // 单遍扫描子块元素：命中当前批（r>>12==b，批=16窗×256行=4096 行 2 的幂）的元素
    // SetValue 聚合至窗口 tile 的 (r&4095)×C+(c−cLo) 位置，并按 occ 首触哨兵去重追加
    // occList（u16 打包批内行号 r&4095，紧凑 flush 免逐行哨兵扫描）；occCnt_ 为批内
    // 占用行数（flush 上界 ≤4096）。内层 tile 粒度仍为 tileElems，MTE2 双缓冲（预取
    // 下一 tile 前锁定当前 tile 元素数）
    __aicore__ inline void BucketScanFillBatch(LocalTensor<Chan> &tile, LocalTensor<uint16_t> &occ,
        LocalTensor<uint16_t> &occList, int64_t cLo, int64_t cHi, int64_t scLo, int64_t scHi,
        uint32_t b)
    {
        const int64_t cSpan = cHi - cLo;
        int64_t colCursor = 0;  // 子块列游标（offBuf 差值展开，只增不减）
        // 当前列段右边界缓存（off(cLo+colCursor+1)−anchor）：仅游标推进时回读 offBuf，
        // 避免逐元素 UB 标量读（游标增量推进的寄存器化实现）
        int64_t colBoundary = (cSpan > 0)
            ? static_cast<int64_t>(offLocal_.GetValue(static_cast<uint32_t>(cLo + 1 - segBase_))) - anchor_
            : 0;
        int64_t tCur = scLo;
        uint32_t nCur = BucketTileSpan(tCur, scHi);
        CopyInTile(tCur, nCur);  // 预取首 tile
        while (tCur < scHi) {
            const int64_t tNext = tCur + nCur;
            uint32_t nNext = 0;
            if (tNext < scHi) {
                nNext = BucketTileSpan(tNext, scHi);
                CopyInTile(tNext, nNext);  // 预取下一 tile，与当前 tile 标量处理重叠
            }
            valBuf_ = valQue_.DeQue<Chan>();
            secBuf_ = secQue_.DeQue<int32_t>();
            for (uint32_t j = 0; j < nCur; j++) {
                const int64_t gj = tCur + j;
                // 段定位（anchor 锚定差值展开，限子块内；offsets 为长度语义不减 base）
                while (colCursor + 1 < cSpan && gj >= colBoundary) {
                    colCursor++;
                    colBoundary = static_cast<int64_t>(offLocal_.GetValue(static_cast<uint32_t>(
                        cLo + colCursor + 1 - segBase_))) - anchor_;
                }
                // CSC：段号 = 列（子块内游标 cRel），secondary = 行；
                // CSR：段号 = 行，secondary = 列。
                // 批维号 secIdx = secondary − indexBase（CSC×ROW 为行号 / CSR×COL 为列号）
                const int64_t secIdx = static_cast<int64_t>(secBuf_.GetValue(j)) - indexBase_;
                // 命中当前批：secIdx>>12 == b（批基 = b×4096）；批内号 = secIdx&4095（全位移）
                if ((static_cast<uint64_t>(secIdx) >> kBatchLog) == static_cast<uint64_t>(b)) {
                    const uint32_t idx = static_cast<uint32_t>(secIdx) & (kBatchRows - 1U);
                    // occ 首触哨兵去重：occ[idx]==0 → 置位并追加 occList（每占用行恰追加一次）
                    if (occ.GetValue(idx) == 0U) {
                        occ.SetValue(idx, 1U);
                        occList.SetValue(occCnt_, static_cast<uint16_t>(idx));
                        occCnt_++;
                    }
                    // 全 dtype 桶填充 SetValue（Copy<int8_t> 对 8 位类型静默失效）
                    tile.SetValue(idx * kColChunk + static_cast<uint32_t>(colCursor),
                        valBuf_.GetValue(j * kUnitsPerElem));
                }
            }
            valQue_.FreeTensor(valBuf_);
            secQue_.FreeTensor(secBuf_);
            tCur = tNext;
            nCur = nNext;
        }
    }

    // occList 紧凑 flush：每占用批维号一次常长段写（满子块恒 32B = C×elemSize；尾子块按
    // (cHi−cLo)×elemSize 收窄——不越出本块段区间 [cLo,cHi)，避免对相邻块段域的 +0 覆写与
    // 其他核产生写竞争）。UB 源 32B 对齐：段源偏移 = idx×C×elemSize = idx×32B（窗口
    // tile 32B 对齐基址 + 32B 整数倍）；GM 目的任意字节地址（DataCopyPad 承接）；
    // blockLen ≤ 32B ≪ 65535B 上界；段内 gap 为窗口清零的 +0（与清零 kernel 幂等，不覆写任何
    // values）。批维号 = 批基 wBatch0 + idx（CSC×ROW 为行号 / CSR×COL 为列号），
    // GM 目的 = dense + 批维号×ld + cLo（cLo < cHi ≤ 段数 ≤ ld，寻址界内：
    // ROW 行×ld+列 / COL 列×ld+行，同一表达式）。
    __aicore__ inline void BucketFlushOccList(LocalTensor<Chan> &tile,
        LocalTensor<uint16_t> &occList, int64_t cLo, int64_t cHi, int64_t wBatch0)
    {
        const uint32_t segBytes =
            static_cast<uint32_t>(cHi - cLo) * kElemBytes;  // 满子块 = 32B（恒长），尾子块收窄
        for (uint32_t i = 0; i < occCnt_; i++) {
            const uint32_t idx = static_cast<uint32_t>(occList.GetValue(i));
            const uint64_t gmElem = (static_cast<uint64_t>(wBatch0) + idx) *
                static_cast<uint64_t>(ld_) + static_cast<uint64_t>(cLo);
            DataCopyPad(denseGm_[gmElem * kUnitsPerElem], tile[idx * kColChunk],
                {1, segBytes, 0, 0, 0});
        }
    }

    __aicore__ inline uint32_t BucketTileSpan(int64_t tBegin, int64_t scHi) const
    {
        int64_t span = scHi - tBegin;  // 尾元素 tile 按实际元素数承接
        if (span > static_cast<int64_t>(tileElems_)) {
            span = static_cast<int64_t>(tileElems_);
        }
        return static_cast<uint32_t>(span);
    }

    // =========================================================================
    // 直写快路径：转置组合（CSC×ROW / CSR×COL）批量散布直写层（仅承接桶门排除域
    // est>238ns；GM 寻址 (secIdx×ld+segIdx)×elemSize）
    //
    // 段块主轴不变（块 b 覆盖段 [b×segsPerBlock, ...)，CSC 段=列/CSR 段=行，
    // grid-stride 认领，块 offsets 区间一次读入复用，≤ segsPerBlock+1 项确定上界）；块内按
    // tileElems 元素 tile 两级循环，两阶段 + 屏障对：
    //   阶段 A（MTE2 + Scalar，V1PackEmitTile 前半）：MTE2 双缓冲预取读入 (secondary,
    //     values) 分片（预取前锁定本 tile 元素数，
    //     与桶路径同款预取式双缓冲）→ 段号按段摊销解码（段游标 v1SegCursor_ 沿 offBuf 单调推进，
    //     每跨一列段解码一次列号 c=段号，段内元素共享，替代基线逐元素列解码——收益主因
    //     之一）→ 每元素 Scalar 打包至 32B 槽位：[value@w0（dtype 通道 SetValue），
    //     gmOff 64 位低/高 32 位@w1–w2（int32 SetValue）]——gmOff=(r×ld+c)×elemSize
    //     （64 位，rows·ld·elemSize 可达 2⁶²，uint64 承接）；
    //   屏障对（每 tile 一次，无逐元素屏障）：PipeBarrier<PIPE_ALL>（Scalar 打包 →
    //     MTE3 读可见性）；
    //   阶段 B（MTE3，V1PackEmitTile 后半）：按槽序每元素一次单 burst DataCopyPad 直写
    //     dense[(r×ld+c)×elemSize]（blockLen=elemSize ≤ 8B ≪ 65535B 上界、GM 任意字节
    //     地址、UB 源槽基址 32B 对齐）→ PipeBarrier<PIPE_MTE3>（MTE3 完成 → 槽缓冲复用）；
    //     写域恰覆盖坐标位置（CSC×ROW：dense[行×ld+列]；CSR×COL：dense[列×ld+行]）；
    //     双缓冲下下一 tile 打包与当前 tile 发射重叠（tile 级流水）；
    //   段跨 tile 续接：tile 末恰为段起点归属下一段、否则段
    //     延续——列游标摊销不受 tile 切分影响；空列段交集为空自然跳过；空块快速跳过；
    //   不引入合并/多 burst（uniform 合并簿记负收益、多 burst 段写 blockLen=4B 仅首
    //     burst 落盘）→ 直写快路径无合并分支；
    //   写域：每元素恰一次 pad 恰覆盖坐标位置 elemSize 字节、不写任何 gap/padding
    //     （区别于桶路径 +0 gap 重写），未覆盖逻辑位置由清零 kernel 承接；ld padding 不触及
    //     （r∈[0,rows)、c∈[0,cols) 由 Host 坐标校验保证）；
    //   写序：激活纯元数据（Init 单点）+ 固定列段遍历序（列游标沿 colOffsets 单调
    //     推进、段内按存储序）；重复坐标合法（同段按存储序先后 pad，结果取任一写入
    //     值）；无重复坐标时发射序=槽序=存储序 → 同输入 bit-wise 一致。
    // =========================================================================
    __aicore__ inline void ProcessV1()
    {
        for (uint32_t blk = blockIdx_; blk < blockNum_; blk += coreNum_) {
            ProcessV1Block(blk);
        }
    }

    __aicore__ inline void ProcessV1Block(uint32_t blk)
    {
        // 段块主轴与块端点定位同基线 ProcessSegBlock：CSC 段=列 / CSR 段=行，
        // 块元素区间 [eBegin, eEnd) 由块端点 offsets 差值（anchor 锚定）定位
        int64_t s0 = 0;
        int64_t s1 = 0;
        int64_t eBegin = 0;
        int64_t eEnd = 0;
        if (!SegBlockRange(blk, s0, s1, eBegin, eEnd)) {
            return;  // 空块（整块全空段）快速跳过
        }

        // 块 offsets 区间 [s0, s1] 一次性读入 offBuf（≤ segsPerBlock+1 项），
        // 块内全部内层 tile 复用（与基线/桶路径同款，确定上界）
        LoadBlockOffsets(s0, s1);

        // 元素 tile 二级循环（预取式双缓冲；段游标跨内层 tile 单调续接）
        v1SegCursor_ = 0;
        int64_t tCur = eBegin;
        uint32_t nCur = V1TileSpan(tCur, eEnd);
        CopyInTile(tCur, nCur);  // 预取首 tile
        while (tCur < eEnd) {
            const int64_t tNext = tCur + nCur;
            uint32_t nNext = 0;
            if (tNext < eEnd) {
                nNext = V1TileSpan(tNext, eEnd);  // 预取前锁定当前 tile 元素数（CopyInTile(next) 会覆写队列状态）
                CopyInTile(tNext, nNext);         // 预取下一 tile，与当前 tile Scalar 打包重叠
            }
            valBuf_ = valQue_.DeQue<Chan>();
            secBuf_ = secQue_.DeQue<int32_t>();
            V1PackEmitTile(tCur, nCur);
            valQue_.FreeTensor(valBuf_);
            secQue_.FreeTensor(secBuf_);
            tCur = tNext;
            nCur = nNext;
        }
        offQue_.FreeTensor(offLocal_);
    }

    __aicore__ inline uint32_t V1TileSpan(int64_t tBegin, int64_t eEnd) const
    {
        int64_t span = eEnd - tBegin;  // 尾 tile 按实际元素数承接（DataCopyPad 字节粒度天然承接）
        if (span > static_cast<int64_t>(tileElems_)) {
            span = static_cast<int64_t>(tileElems_);
        }
        return static_cast<uint32_t>(span);
    }

    // 单 tile：阶段 A 列段分组摊销打包 → PIPE_ALL → 阶段 B 按槽序逐元素单 burst pad
    // 直写 → PIPE_MTE3（屏障对每 tile 一次）
    __aicore__ inline void V1PackEmitTile(int64_t tBegin, uint32_t n)
    {
        LocalTensor<Chan> pack = packSlotQue_.AllocTensor<Chan>();
        LocalTensor<int32_t> packI = pack.template ReinterpretCast<int32_t>();
        LocalTensor<int8_t> packB = pack.template ReinterpretCast<int8_t>();
        const int64_t tileEnd = tBegin + static_cast<int64_t>(n);
        int64_t j = v1SegCursor_;  // 段游标（块内相对段号，CSC 为列号 / CSR 为行号，跨 tile 单调续接）
        int64_t k = 0;             // tile 内元素序 = 槽序（存储序，确定性依据 ②）
        while (k < n) {
            // 当前列段 [pStart, pEnd) 与 tile 区间交集（空列段交集为空自然跳过）
            int64_t pStart = static_cast<int64_t>(
                offLocal_.GetValue(static_cast<uint32_t>(j))) - anchor_;
            if (pStart < tBegin) {
                pStart = tBegin;
            }
            int64_t pEnd = static_cast<int64_t>(
                offLocal_.GetValue(static_cast<uint32_t>(j + 1))) - anchor_;
            if (pEnd > tileEnd) {
                pEnd = tileEnd;
            }
            // 段号按段摊销解码，每段一次（CSC：段号=列 / CSR：段号=行；
            // 段号 < segNum ≤ ld，寻址界内）
            const int64_t segIdx = static_cast<int64_t>(segBase_) + j;
            for (int64_t i = pStart; i < pEnd; i++, k++) {
                const uint32_t e = static_cast<uint32_t>(i - tBegin);
                // 批维号 secIdx = secondary − indexBase（CSC×ROW 为行 / CSR×COL 为列）
                const int64_t secIdx = static_cast<int64_t>(secBuf_.GetValue(e)) - indexBase_;
                // 槽布局 32B（int32 单元视角）：[value@u0（4B 及以下 dtype，dtype 通道
                // SetValue，纯 bit 搬运）/ value lo/hi@u0–u1（complex64 8B），
                // gmOff 64 位 lo/hi@kSlotOffUnit–kSlotOffUnit+1（int32 SetValue，bit 模式
                // 存取）]。GM 寻址：gmOff=(secIdx×ld+segIdx)×elemSize——
                // CSC×ROW 为 (行×ld+列)、CSR×COL 为 (列×ld+行)，同一表达式（转置对称）。
                pack.SetValue(e * kSlotUnits, valBuf_.GetValue(e * kUnitsPerElem));
                if constexpr (kUnitsPerElem == 2U) {
                    // complex64 8B 值：实/虚 2×uint32 占 u0–u1（qNaN/sNaN payload 纯 bit 搬运）
                    pack.SetValue(e * kSlotUnits + 1U, valBuf_.GetValue(e * kUnitsPerElem + 1U));
                }
                const uint64_t gmOff = (static_cast<uint64_t>(secIdx) * static_cast<uint64_t>(ld_) +
                    static_cast<uint64_t>(segIdx)) * static_cast<uint64_t>(kElemBytes);
                packI.SetValue(e * 8U + kSlotOffUnit, static_cast<int32_t>(gmOff & 0xFFFFFFFFULL));
                packI.SetValue(e * 8U + kSlotOffUnit + 1U, static_cast<int32_t>(gmOff >> 32U));
            }
            j++;
        }
        // 屏障对 ①：Scalar 打包 → MTE3 读（跨流水可见性）
        PipeBarrier<PIPE_ALL>();
        // 阶段 B：按槽序每元素一次单 burst pad 直写（blockLen=elemSize，gmOff 64 位
        // lo/hi 重组；UB 源槽基址 32B 对齐，GM 任意字节地址）
        for (uint32_t s = 0; s < n; s++) {
            const uint64_t gmOff = static_cast<uint32_t>(packI.GetValue(s * 8U + kSlotOffUnit)) |
                (static_cast<uint64_t>(static_cast<uint32_t>(packI.GetValue(s * 8U + kSlotOffUnit + 1U))) << 32U);
            DataCopyPad(denseByteGm_[gmOff], packB[s * kSlotBytes], {1, kElemBytes, 0, 0, 0});
        }
        // 屏障对 ②：MTE3 完成 → 槽缓冲复用（双缓冲下下一 tile 打包与本次发射重叠）
        PipeBarrier<PIPE_MTE3>();
        packSlotQue_.FreeTensor(pack);
        // 段游标续接（段跨 tile 切分语义）：tile 末恰为段起点
        // 归属下一段、否则段延续（j 已指向最后处理段的下一段）
        v1SegCursor_ = (tileEnd == static_cast<int64_t>(
            offLocal_.GetValue(static_cast<uint32_t>(j))) - anchor_) ? j : (j - 1);
    }

    // ---- 坐标解析 + 寻址（纯标量路径，仅索引面不承载数据面）----
    // CSR：行号 = 段号（anchor 锚定差值展开，对应公式 np.repeat(np.arange, seg)），
    //      列 = secondary − indexBase；
    // CSC：列号 = 段号，行 = secondary − indexBase（段号为列、secondary 为行）；
    // COO：行 = primary − indexBase，列 = secondary − indexBase。
    // 寻址：ROW B[r,c]=dense[r*ld+c]；COL B[r,c]=dense[c*ld+r]
    // 返回逻辑元素偏移（dtype 元素单位）；GM/通道寻址由 WriteRun 按 kUnitsPerElem 换算。
    __aicore__ inline int64_t TargetOff(int64_t gj)
    {
        int64_t row;
        int64_t col;
        if (format_ == SPARSE2DENSE_FMT_COO) {
            uint32_t i = static_cast<uint32_t>(gj);
            row = static_cast<int64_t>(primBuf_.GetValue(i)) - indexBase_;
            col = static_cast<int64_t>(secBuf_.GetValue(i)) - indexBase_;
        } else {
            // 段定位：offsets[s] − anchor ≤ gj < offsets[s+1] − anchor（anchor 锚定差值展开；
            // 禁止用 raw offsets 与元素下标直接比较，base=1 时会整体 off-by-one）
            while ((segCursor_ + 1) < static_cast<int64_t>(offCount_) &&
                gj >= static_cast<int64_t>(offLocal_.GetValue(static_cast<uint32_t>(segCursor_ + 1))) - anchor_) {
                segCursor_++;
            }
            const int64_t seg = segBase_ + segCursor_;
            const int64_t sec = static_cast<int64_t>(secBuf_.GetValue(static_cast<uint32_t>(gj - tileBegin_))) - indexBase_;
            if (format_ == SPARSE2DENSE_FMT_CSR) {
                row = seg;  // CSR：段号 = 行
                col = sec;  // secondary = 列
            } else {        // CSC：段号 = 列，secondary = 行
                row = sec;
                col = seg;
            }
        }
        return isColMajor_ ? (col * ld_ + row) : (row * ld_ + col);
    }

    // ---- 连续 run 识别 + 打包 + 每 run 一次 DataCopyPad 写出 ----
    // 相邻元素目标 GM 元素地址连续 → run；run 不跨内层 tile（禁跨 tile/跨块合并，
    // 避免写序依赖）；重复坐标合法，结果取任一写入值（同核顺序写/跨核并发写）。
    __aicore__ inline void ComputeAndWrite(int64_t tileBegin, int64_t n)
    {
        tileBegin_ = tileBegin;
        int64_t runStart = 0;
        int64_t runStartOff = TargetOff(tileBegin);
        int64_t prevOff = runStartOff;
        for (int64_t j = 1; j <= n; j++) {
            bool isEnd = (j == n);
            int64_t curOff = 0;
            bool contiguous = false;
            if (!isEnd) {
                curOff = TargetOff(tileBegin + j);
                contiguous = (curOff == prevOff + 1);
            }
            if (isEnd || !contiguous) {
                WriteRun(runStartOff, static_cast<uint32_t>(runStart), static_cast<uint32_t>(j - runStart));
                runStart = j;
                if (!isEnd) {
                    runStartOff = curOff;
                }
            }
            if (!isEnd) {
                prevOff = curOff;
            }
        }
    }

    // 单 run 写出（通道宽度按 V_KIND 编译期选择，纯 bit 搬运，NaN/±inf/±0 bit 保持）：
    //   ① Copy 快速路径（仅 kCopyFast=true 的 4B 通道：TK3/TK4）：run>1 且源 32B 对齐，
    //      Copy 分块 ≤ S2D_COPY_MASK_MAX，PIPE_V + EnQue/DeQue
    //      + PIPE_MTE3；
    //   ② SetValue 逐单元打包（1B/2B 通道主路径 + 4B 非对齐兜底）：Scalar pipe 写 pack 后
    //      MTE3 读存在跨流水可见性要求，PipeBarrier<PIPE_ALL> 前后各一次
    //      （PIPE_V / TQue 事件对 1B/2B 通道不足，PIPE_ALL 排空后 DataCopyPad 可见）。
    __aicore__ inline void WriteRun(int64_t off0, uint32_t srcIdx, uint32_t runLen)
    {
        const uint32_t srcU = srcIdx * kUnitsPerElem;
        const uint32_t units = runLen * kUnitsPerElem;
        const uint64_t gmOff = static_cast<uint64_t>(off0) * kUnitsPerElem;
        const uint32_t runBytes = units * static_cast<uint32_t>(sizeof(Chan));
        if constexpr (kCopyFast) {
            if (runLen > 1U && (srcU & (kCopyAlignUnits - 1U)) == 0U) {
                LocalTensor<Chan> pack = packQue_.AllocTensor<Chan>();
                uint32_t remaining = units;
                uint32_t s = srcU;
                uint32_t d = 0;
                while (remaining > 0U) {
                    uint32_t chunk = (remaining > S2D_COPY_MASK_MAX) ? S2D_COPY_MASK_MAX : remaining;
                    Copy<Chan>(pack[d], valBuf_[s], static_cast<uint64_t>(chunk), 1, {1, 1, 8, 8});
                    remaining -= chunk;
                    s += chunk;
                    d += chunk;
                }
                PipeBarrier<PIPE_V>();
                packQue_.EnQue<Chan>(pack);
                LocalTensor<Chan> packDeq = packQue_.DeQue<Chan>();
                DataCopyPad(denseGm_[gmOff], packDeq, {1, runBytes, 0, 0, 0});
                PipeBarrier<PIPE_MTE3>();
                packQue_.FreeTensor(packDeq);
                return;
            }
        }
        LocalTensor<Chan> pack = packQue_.AllocTensor<Chan>();
        for (uint32_t j = 0; j < units; j++) {
            pack.SetValue(j, valBuf_.GetValue(srcU + j));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(denseGm_[gmOff], pack, {1, runBytes, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        packQue_.FreeTensor(pack);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, SPARSE2DENSE_BUFFER_NUM> valQue_;
    TQue<TPosition::VECIN, SPARSE2DENSE_BUFFER_NUM> secQue_;
    TQue<TPosition::VECIN, SPARSE2DENSE_BUFFER_NUM> primQue_;  // COO 逐点分片
    TQue<TPosition::VECIN, 1> offQue_;                         // CSR/CSC 块 offsets 区间
    TQue<TPosition::VECOUT, SPARSE2DENSE_BUFFER_NUM> packQue_;
    // 直写快路径实例 32B 槽位打包（与 packQue_ 运行时互斥分配，InitUb 按 tiling 分支二选一；
    // 槽距恒 32B 全 dtype 统一，槽内 [value@w0, gmOff 64 位 lo/hi@w1–w2]）
    TQue<TPosition::VECOUT, SPARSE2DENSE_BUFFER_NUM> packSlotQue_;
    // 桶转置实例（与 packQue_ 运行时互斥分配，InitUb 按 tiling 分支二选一）
    TBuf<TPosition::VECCALC> bucketTbuf_;  // 窗口批 tile（W_CO=16 窗 × 8KB 窗口单元 = 128KB）
    TBuf<TPosition::VECCALC> occTbuf_;     // occ 首触哨兵 [4096]（u16，0=未触，批首清零）
    TBuf<TPosition::VECCALC> listTbuf_;    // occList 紧凑占用表 [4096]（u16 打包批内行号 r&4095）

    GlobalTensor<Chan> valGm_;
    GlobalTensor<int32_t> primGm_;
    GlobalTensor<int32_t> secGm_;
    GlobalTensor<int32_t> offGm_;  // CSR/CSC：offsets GM 标量读（anchor/块端点）
    GlobalTensor<Chan> denseGm_;
    GlobalTensor<int8_t> denseByteGm_;  // 直写快路径 dense 字节视图（gmOff 64 位字节偏移）

    LocalTensor<Chan> valBuf_;
    LocalTensor<int32_t> primBuf_;
    LocalTensor<int32_t> secBuf_;
    LocalTensor<int32_t> offLocal_;

    int64_t ld_;
    int64_t nnz_;
    int64_t lenPrimary_;
    int64_t rows_;       // 行数（tiling->rows；转置组合窗口批维 winDim_ 的 ROW 布局取值）
    int64_t cols_;       // 列数（tiling->cols；winDim_ 的 COL 布局取值）
    uint32_t segNum_;
    uint32_t segsPerBlock_;
    uint32_t tileElems_;
    uint32_t blockNum_;
    bool bucketActive_;  // 桶层联合成本门（转置组合×非complex64×est≤238ns）
    bool v1Active_;      // 直写快路径层激活（转置组合×非complex64×桶门排除）
    bool transposed_;    // 转置组合：CSC×ROW 或 CSR×COL（桶转置/直写快路径激活域）
    int64_t winDim_;     // 窗口批维尺寸：ROW 布局取 rows、COL 布局取 cols（门/批数判定）
    int64_t v1SegCursor_;  // 直写快路径段游标（块内相对段号，沿 offBuf 每段解码一次段号，跨 tile 续接）
    uint32_t winBatches_;  // 窗口批数 W_eff = ceil(ceil(rows/256)/16)（重扫遍数，门内 ≤ 小常数）
    uint32_t occCnt_;      // 当前批 occList 占用行数（flush 上界 ≤ kBatchRows=4096）
    int32_t anchor_;     // anchor = offsets[0]（锚定点，CSR/CSC 用）
    uint32_t segBase_;   // 当前块起始段号 s0
    uint32_t offCount_;  // 当前块 offsets 区间项数
    int64_t segCursor_;  // 段扫描游标（offLocal_ 下标）
    int64_t tileBegin_;  // 当前 tile 起始全局元素号
    uint8_t format_;
    int64_t indexBase_;
    uint8_t isColMajor_;
    uint32_t blockIdx_;
    uint32_t coreNum_;
};

// ===========================================================================
// Kernel 入口（逐 TilingKey 实例宏展开：非模板 __global__ 内核）。
// 主线 sparse 库 CMake（仓顶 build.sh --ops=sparse2dense）为单遍 ASC 编译，不支持
// 「模板内核 + 显式实例化」组织形态（ccec 对 __global__ 模板做签名变换，显式实例化
// 语句参数列表无法匹配，报 explicit instantiation does not refer to
// a function template）；故以 GENERATE_S2D_*_KERNEL 宏逐变体展开，设备流（ascendc_kernel_cmake）
// 与主线单遍 ASC 流均直接编译。
// Host 侧按 TilingKey（V_KIND = TK0~TK4）分发到对应后缀入口（见 sparse2dense_host.asc
// / torch_adapter.cpp 的 aclrtlaunch_* 调用点）。禁用废弃宏 TILING_KEY_IS /
// BEGIN_TILING_DATA_DEF。
// ===========================================================================
// ccec 对无 extern "C" 的核函数发出「kernel type ... not marked, auto type derivate
// may be failed」良性告警（ccec 无法静态标注核类型，不影响 launch 与结果）。核类型
// 显式标注 __vector__ 可将告警清零，但其预处理展开 __attribute__((aiv)) 会破坏
// CANN 9.1.0 legacy extract_host_stub.py 的函数签名文法（FUNC_SIGNATURE 仅接受
// cce_kernel + cce_aicore + 签名体，签名体内前置 attribute 使 FUNC_PARAMS 误匹配
// → FuncNameNotFound → libsparse2dense_kernels.a 构建失败），ST Real 直调链路随之
// 失效，故不采用，保留良性告警。
// ===========================================================================
#define GENERATE_S2D_ZERO_KERNEL(V_KIND, SUFFIX)                                            \
    __global__ __aicore__ void sparse2dense_zero_kernel_##SUFFIX(GM_ADDR denseGm,           \
        const Sparse2DenseTilingData tiling)                                                \
    {                                                                                       \
        KernelS2DZero<V_KIND> op;                                                           \
        op.Init(denseGm, tiling);                                                           \
        op.Process();                                                                       \
    }

#define GENERATE_S2D_SCATTER_KERNEL(V_KIND, SUFFIX)                                         \
    __global__ __aicore__ void sparse2dense_scatter_kernel_##SUFFIX(GM_ADDR valuesGm,       \
        GM_ADDR primaryGm, GM_ADDR secondaryGm, GM_ADDR denseGm,                            \
        const Sparse2DenseTilingData tiling)                                                \
    {                                                                                       \
        KernelS2DScatter<V_KIND> op;                                                        \
        op.Init(valuesGm, primaryGm, secondaryGm, denseGm, tiling);                         \
        op.Process();                                                                       \
    }

// TK0~TK4 全 dtype（支持的全部 dtype 组合；
// 后缀命名即 host 侧 TilingKey 分发目标入口名）
GENERATE_S2D_ZERO_KERNEL(SPARSE2DENSE_VK_INT8, int8)      // TK0：uint16 通道造零（覆盖 2×int8）
GENERATE_S2D_ZERO_KERNEL(SPARSE2DENSE_VK_FLOAT16, fp16)   // TK1：half 通道
GENERATE_S2D_ZERO_KERNEL(SPARSE2DENSE_VK_BFLOAT16, bf16)  // TK2：bfloat16_t 通道
GENERATE_S2D_ZERO_KERNEL(SPARSE2DENSE_VK_FLOAT32, fp32)   // TK3：float 通道（主路径）
GENERATE_S2D_ZERO_KERNEL(SPARSE2DENSE_VK_COMPLEX64, c64)  // TK4：uint32 通道（2×float32）

GENERATE_S2D_SCATTER_KERNEL(SPARSE2DENSE_VK_INT8, int8)     // TK0：int8 通道（SetValue 打包 + PIPE_ALL）
GENERATE_S2D_SCATTER_KERNEL(SPARSE2DENSE_VK_FLOAT16, fp16)  // TK1：uint16 通道（2B）
GENERATE_S2D_SCATTER_KERNEL(SPARSE2DENSE_VK_BFLOAT16, bf16) // TK2：uint16 通道（2B）
GENERATE_S2D_SCATTER_KERNEL(SPARSE2DENSE_VK_FLOAT32, fp32)  // TK3：uint32 通道（主路径）
GENERATE_S2D_SCATTER_KERNEL(SPARSE2DENSE_VK_COMPLEX64, c64) // TK4：uint32 通道 ×2 单元/元素

#undef GENERATE_S2D_SCATTER_KERNEL
#undef GENERATE_S2D_ZERO_KERNEL

// ============================================================================
// 坐标校验链设备侧 hash 守卫内核——内容指纹缓存守卫。
//
// 目的：坐标校验链结果记忆化的内容守卫。缓存命中必须以设备侧内容指纹为前提——仅按
// 指针 + 规格命中时，调用方原地改写索引数组（指针/规格不变）会复用陈旧 SUCCESS，
// 新的越界坐标将直接进入 scatter 越界写 matB。本内核对
// primary/secondary 全量元素采集 FNV-1a 指纹（O(nnz) 设备并行），Host 侧以
// 「指针 + 规格 + 指纹 + numBlocks」比对缓存：同内容同指纹 → 命中返回缓存结论；
// 内容变化（含原地改写）→ 指纹必变 → 未命中回退全量坐标重校（语义不变）。
//
// 向量化指纹（64-lane 并行 FNV-1a 32 双链）：AIV 无 64 位乘法向量通路，逐元素标量
// FNV-1a 64 需数百 µs/百万元素级（大 nnz 用例性能预算不可承受）。本内核按 64 lane
//（256B/4B）把元素流切成 64 条独立 FNV-1a 32 位链，双链（A/B 独立 offset + prime）
// 折叠恢复 64 位碰撞强度（单元素变更漏检 ~2^-64）；每 64 元素仅 4 条向量指令
//（Xor int16 视图 ×2 + Mul int32 ×2），指纹采集为带宽受限（MB 级索引 ~10µs 量级）。
//
// 确定性与灵敏度：元素到 lane 的映射由 (blockIdx, numBlocks, CHUNK_ELEMS) 静态划分
// 决定（numBlocks 在缓存键内），与核数无关地可复现；任一元素变化至少扰动一条 lane
// 链，lane 间与块间经 Host FoldPartials 按块序折叠，次序敏感。指纹仅作缓存守卫，
// 校验语义仍由全量坐标校验承接。
//
// 协议：
//   入参 primaryGm/secondaryGm：int32 数组（长度为空时传非空占位指针，本内核按长度守卫不访存）
//   入参 params（按值下发，清零/scatter kernel tiling 同款形态）：S2DF3HashParams { lenPrimary, nnz, nonce }
//   出参 outGm：uint64 partials[3 * numBlocks]，blockIdx b 写 [3b]=primary 指纹、
//   [3b+1]=secondary 指纹、[3b+2]=nonce 盐（空数组指纹 = 全 offset basis 链的折叠值）。
//   Host 侧逐块比对盐位后按块序折叠（(h ^ partial) * prime 串联）得全量指纹与缓存比对：
//   发射未运行/半途残留的陈旧 partials 盐位必失配（nonce 严格递增不重复）→ 转 cache
//   miss → 全量坐标重校（见 sparse2dense_tiling_data.h）。
// ============================================================================

namespace {

constexpr uint32_t CHUNK_ELEMS = 2048;  // 8KB UB 分块（MTE2 单次 DataCopyPad 字节粒度承接尾块）
constexpr uint32_t LANE_ELEMS = 64;     // 向量通道数（256B / 4B）：每 lane 一条 FNV-1a 32 位链
constexpr int32_t FNV32_OFFSET_A = static_cast<int32_t>(0x811C9DC5U);  // FNV-1a 32 offset basis（链 A）
constexpr int32_t FNV32_OFFSET_B = static_cast<int32_t>(0xC2B2AE35U);  // 链 B 独立 offset
constexpr int32_t FNV32_PRIME_A = 16777619;                             // 0x01000193 FNV-1a 32 标准 prime（链 A）
constexpr int32_t FNV32_PRIME_B = static_cast<int32_t>(0x9E3779B1U);   // 链 B 独立奇素数
constexpr uint64_t FNV64_OFFSET_BASIS = 1469598103934665603ULL;
constexpr uint64_t FNV64_PRIME = 1099511628211ULL;

// 单块单数组的 lane 链终折叠：lane 序（先链 A 后链 B）FNV-1a 64 串联。空数组的折叠值
// 由 Duplicate 初始化的 offset basis 链唯一确定（确定性守卫）。
__aicore__ inline uint64_t FoldLanes(const LocalTensor<int32_t> &accA, const LocalTensor<int32_t> &accB)
{
    uint64_t hash = FNV64_OFFSET_BASIS;
    for (uint32_t lane = 0; lane < LANE_ELEMS; lane++) {
        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(accA.GetValue(lane)));
        hash *= FNV64_PRIME;
    }
    for (uint32_t lane = 0; lane < LANE_ELEMS; lane++) {
        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(accB.GetValue(lane)));
        hash *= FNV64_PRIME;
    }
    return hash;
}

// 块内 64-lane 向量混链：元素到 lane 的映射 = 块内偏移 mod LANE_ELEMS；count 掩码承接
// 尾向量（不足 64 的尾部按实际元素数掩码混入，lane 链确定性收短）。向量 Xor 仅支持
// 16 位 dtype——int16 视图逐半字异或与 32 位按位异或逐位等价，无信息损失；Mul 为
// int32 元素级（模 2^32 环绕，FNV-1a 32 单步 h = (h ^ v) * prime）。
__aicore__ inline void MixChunkLanes(const LocalTensor<int32_t> &accA, const LocalTensor<int32_t> &accB,
    const LocalTensor<int32_t> &primeA, const LocalTensor<int32_t> &primeB, const LocalTensor<int32_t> &chunk,
    uint32_t count)
{
    const LocalTensor<int16_t> accA16 = accA.ReinterpretCast<int16_t>();
    const LocalTensor<int16_t> accB16 = accB.ReinterpretCast<int16_t>();
    const LocalTensor<int16_t> chunk16 = chunk.ReinterpretCast<int16_t>();
    for (uint32_t off = 0; off < count; off += LANE_ELEMS) {
        const uint32_t laneCount = (count - off) < LANE_ELEMS ? (count - off) : LANE_ELEMS;
        Xor(accA16, accA16, chunk16[off * 2U], laneCount * 2U);
        Mul(accA, accA, primeA, static_cast<int32_t>(laneCount));
        Xor(accB16, accB16, chunk16[off * 2U], laneCount * 2U);
        Mul(accB, accB, primeB, static_cast<int32_t>(laneCount));
    }
}

// 分块 strided 扫描 + lane 链混入：block b 自 b*CHUNK_ELEMS 起以 numBlocks*CHUNK_ELEMS
// 步进覆盖 [0, len)，块内按 64 元素向量步进混链；扫描完毕折叠为本块指纹。
__aicore__ inline uint64_t HashStridedLanes(TQue<TPosition::VECIN, 2> &inQue, const GlobalTensor<int32_t> &src,
    uint64_t len, uint32_t blockIdx, uint32_t numBlocks, const LocalTensor<int32_t> &accA,
    const LocalTensor<int32_t> &accB, const LocalTensor<int32_t> &primeA, const LocalTensor<int32_t> &primeB)
{
    Duplicate(accA, FNV32_OFFSET_A, LANE_ELEMS);
    Duplicate(accB, FNV32_OFFSET_B, LANE_ELEMS);
    for (uint64_t begin = static_cast<uint64_t>(blockIdx) * CHUNK_ELEMS; begin < len;
        begin += static_cast<uint64_t>(numBlocks) * CHUNK_ELEMS) {
        const uint64_t remain = len - begin;
        const uint32_t count = static_cast<uint32_t>(remain < CHUNK_ELEMS ? remain : CHUNK_ELEMS);
        LocalTensor<int32_t> ub = inQue.AllocTensor<int32_t>();
        DataCopyPad(ub, src[begin], {1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        inQue.EnQue<int32_t>(ub);
        LocalTensor<int32_t> ubDeq = inQue.DeQue<int32_t>();
        MixChunkLanes(accA, accB, primeA, primeB, ubDeq, count);
        inQue.FreeTensor(ubDeq);
    }
    return FoldLanes(accA, accB);
}

}  // namespace

// 非模板 __global__ 内核（params 按值随 <<<>>> 下发，
// 见上方协议节；stub 同名同参，见 sparse2dense_host.cpp 调用点）
__global__ __aicore__ void sparse2dense_f3_hash_kernel(GM_ADDR primaryGmAddr, GM_ADDR secondaryGmAddr,
    const S2DF3HashParams params, GM_ADDR outGmAddr)
{
    const uint32_t numBlocks = GetBlockNum();
    const uint32_t blockIdx = GetBlockIdx();

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> accBufA;
    TBuf<TPosition::VECCALC> accBufB;
    TBuf<TPosition::VECCALC> primeBufA;
    TBuf<TPosition::VECCALC> primeBufB;
    pipe.InitBuffer(inQue, 2, CHUNK_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(outQue, 1, 3 * sizeof(uint64_t));
    pipe.InitBuffer(accBufA, LANE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(accBufB, LANE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(primeBufA, LANE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(primeBufB, LANE_ELEMS * sizeof(int32_t));

    const LocalTensor<int32_t> accA = accBufA.Get<int32_t>();
    const LocalTensor<int32_t> accB = accBufB.Get<int32_t>();
    const LocalTensor<int32_t> primeA = primeBufA.Get<int32_t>();
    const LocalTensor<int32_t> primeB = primeBufB.Get<int32_t>();
    Duplicate(primeA, FNV32_PRIME_A, LANE_ELEMS);
    Duplicate(primeB, FNV32_PRIME_B, LANE_ELEMS);

    GlobalTensor<int32_t> primaryGm;
    GlobalTensor<int32_t> secondaryGm;
    primaryGm.SetGlobalBuffer((__gm__ int32_t *)primaryGmAddr);
    secondaryGm.SetGlobalBuffer((__gm__ int32_t *)secondaryGmAddr);

    const uint64_t hashP =
        HashStridedLanes(inQue, primaryGm, params.lenPrimary, blockIdx, numBlocks, accA, accB, primeA, primeB);
    const uint64_t hashS =
        HashStridedLanes(inQue, secondaryGm, params.nnz, blockIdx, numBlocks, accA, accB, primeA, primeB);

    // 部分指纹写出：{hashP, hashS, nonce 盐} 三元组 UB 标量打包 → PIPE_ALL 屏障 →
    // MTE3 DataCopyPad 写 GM（GM 标量写通道不承接；标量写→MTE3 跨流水可见性需
    // 屏障守卫）。盐位由 Host 逐块比对（陈旧 partials 检出，见上方协议节）
    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer((__gm__ uint64_t *)outGmAddr, 3ULL * numBlocks);
    LocalTensor<uint64_t> outUb = outQue.AllocTensor<uint64_t>();
    outUb.SetValue(0, hashP);
    outUb.SetValue(1, hashS);
    outUb.SetValue(2, params.nonce);
    PipeBarrier<PIPE_ALL>();
    outQue.EnQue<uint64_t>(outUb);
    LocalTensor<uint64_t> outDeq = outQue.DeQue<uint64_t>();
    DataCopyPad(outGm[3ULL * blockIdx], outDeq, {1, 3 * sizeof(uint64_t), 0, 0, 0});
    outQue.FreeTensor(outDeq);
}
