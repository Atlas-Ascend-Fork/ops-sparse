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

/*!
 * \file sparse2dense_tiling_data.h
 * \brief aclsparse_sparse2dense TilingData 结构体定义与 TilingKey 模板参数声明（Host / Kernel 共用）。
 *
 * 字段架构与 arch35 tiling 数据（m/n/indexBase/valueType/isColMajor/ld/
 * perBlock/format/nnz）同构，新增 zero 系列与 tileElems/segsPerBlock/blockNum/segNum 支撑 MemBase 分片。
 * 按公共约束不使用废弃宏 BEGIN_TILING_DATA_DEF，采用普通结构体 + REGISTER_TILING_DEFAULT 注册。
 * per-core/per-block 分片不下发（kernel 侧 GetBlockIdx/GetBlockNum grid-stride 动态认领）。
 */

#ifndef SPARSE2DENSE_TILING_DATA_H_
#define SPARSE2DENSE_TILING_DATA_H_

#include <cstdint>

#include "ascendc/host_api/tiling/template_argument.h"

// 稀疏格式编码（TilingData 运行时分支，不进 TilingKey）
constexpr uint8_t SPARSE2DENSE_FMT_CSR = 0;
constexpr uint8_t SPARSE2DENSE_FMT_CSC = 1;
constexpr uint8_t SPARSE2DENSE_FMT_COO = 2;

// valueKind 编码（与 TilingKey V_KIND 一一对应，TK0~TK4）
constexpr uint8_t SPARSE2DENSE_VK_INT8 = 0;      // TK0 int8      1B（清零走 uint16 通道）
constexpr uint8_t SPARSE2DENSE_VK_FLOAT16 = 1;   // TK1 float16   2B
constexpr uint8_t SPARSE2DENSE_VK_BFLOAT16 = 2;  // TK2 bfloat16  2B
constexpr uint8_t SPARSE2DENSE_VK_FLOAT32 = 3;   // TK3 float32   4B
constexpr uint8_t SPARSE2DENSE_VK_COMPLEX64 = 4; // TK4 complex64 8B（2×float32 通道）

// 分片估算基准（NNZ_TILE=1024、SEG_BLOCK_SEGS=64）
constexpr uint32_t SPARSE2DENSE_NNZ_TILE = 1024;
constexpr uint32_t SPARSE2DENSE_SEG_BLOCK_SEGS = 64;
constexpr uint32_t SPARSE2DENSE_BUFFER_NUM = 2;
// 清零 kernel 零块估算基准字节（零块 8KB；维持设计期预算口径、不做运行时收窄，
// GetCoreMemSize 简化项说明见 sparse2dense_kernel.cpp 清零 kernel Init 与 sparse2dense_tiling.cpp 头注）
constexpr uint32_t SPARSE2DENSE_ZERO_TILE_BYTES = 8192;

// 坐标校验链设备侧 hash 守卫内核参数（按值随 <<<>>> 下发，sparse2dense_kernel.cpp 读取；
// 与清零/scatter kernel 的 tiling 按值传参同款形态，免去每调用 16B 级 H2D）
struct S2DF3HashParams {
    uint64_t lenPrimary = 0;  ///< primary 元素数（CSR/CSC=rows/cols+1；COO=nnz）
    uint64_t nnz = 0;         ///< secondary 元素数
    uint64_t nonce = 0;       ///< 本次调用一次性盐（严格递增）：内核将其随块指纹一并写回
                               ///<（partials[3b+2]），Host 逐块比对——<<<>>> 发射无返回值，
                               ///< 未运行/半途残留的陈旧 partials 经盐失配转为 cache miss →
                               ///< 回退全量坐标重校，不误当本次指纹比对
};

// ===========================================================================
// 桶层联合成本门——Host tiling 与 scatter kernel Init 同式纯
// 元数据判定。整数运算（ns 口径），仅依赖 tiling 元数据与
// 核数、不含数据内容 → 同输入同路径同划分（确定性前提）。
//
//   bucketActive = transposed && !complex64 && est ≤ S2D_GATE_BUDGET_NS（=238 = 0.85×基线 280ns）
//   transposed 转置组合：CSC×ROW 或 CSR×COL（secondary 维作窗口批维、段维作列子块维，
//   GM 寻址同为 (sec×ld+seg)×elemSize，桶转置/直写快路径两组合共用同一实现）
//   est = W_eff×S + H + seq×F/elemsPerCore + M
//     W_eff        = ceil(ceil(winDim/rowTile) / W_CO)                         重扫遍数（窗口批数；
//                    winDim = 窗口批维尺寸：ROW 布局取 rows、COL 布局取 cols）
//     seq          = ceil(blockNum×ceil(segsPerBlock/C)×W_eff / activeCores)   每核批数
//     elemsPerCore = ceil(nnz / activeCores)
//     activeCores  = min(coreNum, blockNum)（scatter kernel 侧 GetBlockNum() 即 launch grid=min(blockNum,核数)）
//   性能锚点：S=33ns（Scalar 重扫单价/遍/元素）、H=70ns（命中：SetValue+occ 首触）、
//   F=7μs（批固定：清零+occ 初始化+flush+屏障对）、M=20ns（恒 32B 段写摊销）。
//   W_CO 降档护栏：W_CO 为编译期单点参数（kernel 侧 kWinCo），降档时 bucketTile 预算同比缩小、
//   W_eff 经本式同比增大 → 激活域自动收窄（设计期预算口径，不做运行时 GetCoreMemSize 收窄，
//   与清零/scatter kernel 缓冲预算同口径）。
//
// 编译注记：本头文件同时被 gcc 纯 host 编译（op_host/sparse2dense_tiling.cpp）与 ccec
// ASC 单源编译（sparse2dense_kernel.cpp / sparse2dense_host.asc）包含。ccec 无 host/
// device 双标注（__aicore__ 函数 host 区不可调用、未标注函数 __aicore__ 区不可调用，
// 两方向均被拒绝），故公式核心以宏单点定义（唯一公式文本，防双份漂移），Device /
// Host 两路 wrapper 各自标注——保证「Host tiling 与 scatter kernel Init 同式」。
// ===========================================================================
#define S2D_GATE_S_NS 33ULL        // Scalar 重扫单价（每遍每元素，ns）
#define S2D_GATE_H_NS 70ULL        // 命中（值 SetValue + occ 首触哨兵，ns）
#define S2D_GATE_M_NS 20ULL        // MTE3 恒 32B 常长段写摊销（ns）
#define S2D_GATE_F_NS 7000ULL      // 批固定开销（清零+occ init+flush+屏障对，ns）
#define S2D_GATE_BUDGET_NS 238ULL  // 0.85 × 基线 280ns/元素

// 窗口批数 W_eff = ceil(ceil(rows/rowTile)/W_CO)（rowTile=256、W_CO=16 时单批覆盖 4096 行）
#define S2D_GATE_WIN_BATCHES(rows_, rowTile_, winCo_) \
    (((((rows_) + (rowTile_) - 1) / (rowTile_)) + (winCo_) - 1) / (winCo_))

// est 公式核心（ns；调用方先判 S2D_GATE_COND_CORE 保 ac/wEff 非零，避免除零）
#define S2D_GATE_EST_CORE(nnz_, blockNum_, wEff_, chunksPerBlock_, ac_)                     \
    ((wEff_) * S2D_GATE_S_NS + S2D_GATE_H_NS + S2D_GATE_M_NS +                              \
     ((((blockNum_) * (chunksPerBlock_) * (wEff_) + (ac_) - 1) / (ac_)) * S2D_GATE_F_NS) /  \
         (((nnz_) + (ac_) - 1) / (ac_)))

// 激活条件（est ≤ 预算由两路 wrapper 短路承接；blockNum/coreNum/rowTile/colChunk 非零防除零）
#define S2D_GATE_COND_CORE(valueSupported_, isTransposed_, winDim_, nnz_, blockNum_,        \
    rowTile_, colChunk_, coreNum_)                                                          \
    ((valueSupported_) && (isTransposed_) && (winDim_) > 0 && (nnz_) > 0 &&                 \
     (blockNum_) != 0 && (rowTile_) != 0 && (colChunk_) != 0 && (coreNum_) != 0)

#ifdef __clang__
#if defined(S2D_GATE_HOST_BUILD)
#define S2D_GATE_ATTR  // 非 ccec 的 clang 前端 host 检查覆盖（本仓库构建不使用）
#else
#define S2D_GATE_ATTR __aicore__
#endif
#else
#define S2D_GATE_ATTR
#endif

// ---- Device wrapper（scatter kernel Init 调用；ccec __aicore__ 区）----
S2D_GATE_ATTR constexpr inline uint64_t S2DBucketEstNs(uint64_t winDim, uint64_t nnz,
    uint64_t blockNum, uint64_t segsPerBlock, uint64_t colChunk, uint64_t rowTile, uint64_t winCo,
    uint64_t coreNum)
{
    const uint64_t ac = (coreNum < blockNum) ? coreNum : blockNum;  // activeCores = min
    const uint64_t wEff = S2D_GATE_WIN_BATCHES(winDim, rowTile, winCo);
    if (ac == 0 || wEff == 0 || colChunk == 0) {  // colChunk==0 防除零（调用方门条件已含，双保险）
        return 0;
    }
    return S2D_GATE_EST_CORE(nnz, blockNum, wEff, (segsPerBlock + colChunk - 1) / colChunk, ac);
}

S2D_GATE_ATTR constexpr inline bool S2DBucketGateActive(bool valueSupported, bool isTransposed,
    uint64_t winDim, int64_t nnz, uint64_t blockNum, uint64_t segsPerBlock,
    uint64_t colChunk, uint64_t rowTile, uint64_t winCo, uint64_t coreNum)
{
    if (!S2D_GATE_COND_CORE(valueSupported, isTransposed, winDim, nnz, blockNum, rowTile,
        colChunk, coreNum)) {
        return false;
    }
    return S2DBucketEstNs(winDim, static_cast<uint64_t>(nnz), blockNum,
        segsPerBlock, colChunk, rowTile, winCo, coreNum) <= S2D_GATE_BUDGET_NS;
}

// ---- Host wrapper（Host tiling 自检/镜像调用；gcc 与 ccec host 区均可）----
inline uint64_t S2DBucketEstNsHost(uint64_t winDim, uint64_t nnz, uint64_t blockNum,
    uint64_t segsPerBlock, uint64_t colChunk, uint64_t rowTile, uint64_t winCo, uint64_t coreNum)
{
    const uint64_t ac = (coreNum < blockNum) ? coreNum : blockNum;  // activeCores = min
    const uint64_t wEff = S2D_GATE_WIN_BATCHES(winDim, rowTile, winCo);
    if (ac == 0 || wEff == 0 || colChunk == 0) {  // colChunk==0 防除零（调用方门条件已含，双保险）
        return 0;
    }
    return S2D_GATE_EST_CORE(nnz, blockNum, wEff, (segsPerBlock + colChunk - 1) / colChunk, ac);
}

inline bool S2DBucketGateActiveHost(bool valueSupported, bool isTransposed, uint64_t winDim,
    int64_t nnz, uint64_t blockNum, uint64_t segsPerBlock, uint64_t colChunk,
    uint64_t rowTile, uint64_t winCo, uint64_t coreNum)
{
    if (!S2D_GATE_COND_CORE(valueSupported, isTransposed, winDim, nnz, blockNum, rowTile,
        colChunk, coreNum)) {
        return false;
    }
    return S2DBucketEstNsHost(winDim, static_cast<uint64_t>(nnz), blockNum,
        segsPerBlock, colChunk, rowTile, winCo, coreNum) <= S2D_GATE_BUDGET_NS;
}

#undef S2D_GATE_ATTR

struct Sparse2DenseTilingData {
    int64_t rows = 0;         ///< matA/matB 行数
    int64_t cols = 0;         ///< matA/matB 列数
    int64_t ld = 0;           ///< DnMat leading dimension（ROW: ld>=cols; COL: ld>=rows）
    int64_t nnz = 0;          ///< 非零元个数 = len(values) = len(secondary)
    int64_t lenPrimary = 0;   ///< len(primary)：CSR=rows+1 / CSC=cols+1 / COO=nnz
    int64_t zeroTotalBytes = 0;  ///< 清零 kernel 总字节数 =（ROW?rows:cols)*ld*elemSize（含 ld padding）
    uint32_t zeroTileBytes = SPARSE2DENSE_ZERO_TILE_BYTES;  ///< 清零 kernel 每次零块字节数（UB 预算内）
    uint32_t tileElems = SPARSE2DENSE_NNZ_TILE;     ///< scatter kernel 段内元素 tile 大小（二级轴）
    uint32_t segsPerBlock = SPARSE2DENSE_SEG_BLOCK_SEGS;  ///< scatter kernel 段块主轴粒度；COO 不使用
    uint32_t blockNum = 0;    ///< scatter kernel 块数：CSR/CSC=ceil(segNum/segsPerBlock)；COO=ceil(nnz/tileElems)
    uint32_t segNum = 0;      ///< 段数：CSR=rows / CSC=cols / COO=0（无段头）
    uint8_t format = SPARSE2DENSE_FMT_CSR;  ///< 0=CSR / 1=CSC / 2=COO
    uint8_t indexBase = 0;    ///< 索引 base：0/1（仅坐标减 base；offsets 不减）
    uint8_t isColMajor = 0;   ///< 0=ROW（B[r,c]=dense[r*ld+c]）/ 1=COL（B[r,c]=dense[c*ld+r]）
    uint8_t valueKind = SPARSE2DENSE_VK_FLOAT32;  ///< TK0~TK4 编码（与 TilingKey 一致）
    uint64_t workspaceSize = 0;  ///< 恒 0（bufferSize 恒 0 协议）
};

// ===========================================================================
// TilingKey 模板参数声明。
//
// TilingKey 只有一个轴：dtype（V_KIND，TK0~TK4 与支持的
// 5 个 dtype 组合一一对应）。format/layout/index_base 均为 TilingData 运行时参数，不进 TilingKey。
//
// 按公共约束使用 ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL 模板参数分发（Host 侧配
// ASCENDC_TPL_SEL_PARAM / GET_TPL_TILING_KEY 编译期编码 TilingKey），禁用废弃宏
// TILING_KEY_IS / BEGIN_TILING_DATA_DEF。
//
// complex64（TK4）不使用 ASCENDC_TPL_DATATYPE_DECL：TypeFromId<DT_COMPLEX64> 仅 arch35
// （DAV_3510）可用，arch22（DAV_2201）无原生复数类型，按 2×float32 通道承接，
// 故以 UINT 轴 V_KIND 表达 5 个 dtype 实例。
// ===========================================================================
ASCENDC_TPL_ARGS_DECL(Sparse2Dense,
    ASCENDC_TPL_UINT_DECL(V_KIND, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_KIND, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_KIND, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_KIND, ASCENDC_TPL_UI_LIST, 2)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_KIND, ASCENDC_TPL_UI_LIST, 3)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(V_KIND, ASCENDC_TPL_UI_LIST, 4)),
);

#endif  // SPARSE2DENSE_TILING_DATA_H_
