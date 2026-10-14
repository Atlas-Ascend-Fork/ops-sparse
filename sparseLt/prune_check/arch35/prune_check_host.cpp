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
 * \file prune_check_host.cpp
 * \brief aclsparseLtSpMMAPruneCheck host-side API implementation.
 *
 * 校验 d_in 是否满足 2:4 结构化稀疏条件，结果写入 d_valid（0 满足 / 1 不满足；
 * numBatches>1 时为全部 batch 判定的 OR）。三段式结构：参数校验与派生 →
 * tiling 构造 → d_valid 预清零 + kernel 异步启动；tiling 以 by-value 经
 * kernel launch args 传递（无额外 tilingGm，异步语义）。
 */

// Include order: C standard → C++ → CANN → local.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "shared/aclsparselt_internal.h"
#include "../../../sparse/common/aclsparse_host_utils.h"
#include "prune_check/arch35/prune_check_kernel.h"

namespace {
constexpr const char* kLogTag = "aclsparseLtSpMMAPruneCheck";
constexpr uint64_t kAddressLimit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
// rows/cols/ld 的 32 字节对齐要求按元素字节数换算为元素对齐（结构化描述符 init
// 已保证，此处对 kernel 直接消费的扫描维度做防御性复检，防伪造描述符）。
constexpr int kMatrixAlignmentBytes = 32;
constexpr uint64_t kVectorBytes = 256;
// UB 工作上限（具名常量，来源推导）：平台实际容量经 GetUbSize() 动态查询
//（PlatformAscendCManager::GetCoreMemSize(UB)，本算子与 spmma_compress 共用该
// 查询），kUbLimit 为其上的保守 cap —— 950PR（DAV_3510）AIV 单核 UB 规格
// 256KB，取 256KB−8KB=248KB 为栈/杂项留出余量（与 spmma_compress 同款取值，
// 仓内无 SDK 统一共享常量）。std::min 保证更小 UB 平台自动收缩为平台值。
// 已知限制：后续架构 UB 容量变更时需复核此 cap（放大不改变正确性，仅影响
// 单 tile 搬运粒度）。
constexpr uint64_t kUbReserve = 4096;
constexpr uint64_t kUbLimit = 248 * 1024;
// reduceBuf：anyViolate 寄存器落地槽（一个向量宽度）。
constexpr uint64_t kReduceBufferBytes = kVectorBytes;
// 行带粒度（= prune SPLT_PRUNE_ROW_TILE / spmma_compress 行带粒度）；
// 为 g∈{2,4} 的倍数，保证 ColScan 垂直组不跨行带边界。
constexpr uint32_t kTileRows = 16;
// AIV_ONLY kernel 以 64 AI Vector 为上限。
constexpr uint32_t kMaxBlockDim = 64;

struct PruneCheckParams {
    PruneCheckTilingData tiling;
    uint32_t blockDim = 0;
    const void* input = nullptr;
    int* dValid = nullptr;
};

bool CheckedMulAdd(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t& result)
{
    if (c > kAddressLimit || (b != 0 && a > (kAddressLimit - c) / b)) {
        return false;
    }
    result = a * b + c;
    return true;
}

// ----------------------------------------------------------------------------
// SelectSparseSide — 校验表 #3/#4：matA/matB 非空 + 恰有一个结构化稀疏侧。
// D-1 偏离：prune 对「A 非稀疏」一律走 B-sparse 路径（dense×dense 也静默剪枝 B）；
// 校验算子对无稀疏侧/双稀疏侧输入若静默选取将产出无意义判定的「合法」结论，
// 故显式拒绝（NOT_SUPPORTED）。
// ----------------------------------------------------------------------------
aclsparseStatus_t SelectSparseSide(const aclsparseLtMatmulDescriptor* md,
    const aclsparseLtMatDescriptor*& sparse, bool& sparseA)
{
    if (md->matA == nullptr || md->matB == nullptr) {
        OP_LOGE(kLogTag, "matA/matB is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const bool aStructured = md->matA->isStructured;
    const bool bStructured = md->matB->isStructured;
    if (aStructured == bStructured) {
        OP_LOGE(kLogTag, "exactly one structured sparse side required (A=%d, B=%d)",
                static_cast<int>(aStructured), static_cast<int>(bStructured));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    sparseA = aStructured;
    sparse = aStructured ? md->matA : md->matB;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// DeriveBatchParams — 校验表 #6：批量参数。numBatches < 1 → INVALID_VALUE
//（SetAttribute 已保证 ≥ 1，此处为防伪造描述符的防御复检）；numBatches > 1 时
// batchStride 须 > 0 且为 32B/dataBytes 元素数的整数倍（与 rows/cols/ld 同款
// 对齐：batch 基址偏移 batchStride×dataBytes 须维持 32B 倍数性质，保证各
// batch 的 DataCopyPad 源地址对齐性质与 batch 0 一致）。numBatches == 1 时
// batchStride 不被消费（恒按 0 处理），单 batch 数值路径不变。
// 须在 #5 dtype 白名单之后调用（对齐约束的除数依赖 dataBytes）。
// ----------------------------------------------------------------------------
aclsparseStatus_t DeriveBatchParams(const aclsparseLtMatDescriptor* sparse, uint32_t dataBytes,
    PruneCheckParams& params)
{
    auto& td = params.tiling;
    if (sparse->numBatches < 1) {
        OP_LOGE(kLogTag, "sparse side numBatches=%d < 1", static_cast<int>(sparse->numBatches));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    td.numBatches = static_cast<uint64_t>(sparse->numBatches);
    td.batchStride = 0;
    if (sparse->numBatches > 1) {
        // 防御性复检（不可达：#5 dtype 白名单保证 dataBytes ∈ {4,2,1}，dtype
        // 先行调用序见 DeriveCheckShape），显式校验使下方除法安全可证。
        if (dataBytes == 0) {
            OP_LOGE(kLogTag, "dataBytes unexpectedly zero");
            return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
        }
        const int64_t batchStride = sparse->batchStride;
        const uint64_t batchDivisor = static_cast<uint64_t>(kMatrixAlignmentBytes / dataBytes);
        if (batchStride <= 0 || static_cast<uint64_t>(batchStride) % batchDivisor != 0) {
            OP_LOGE(kLogTag,
                    "invalid batchStride=%lld for numBatches=%d: must be > 0 and a multiple of "
                    "%llu elements (32B/element size)",
                    static_cast<long long>(batchStride),
                    static_cast<int>(sparse->numBatches),
                    static_cast<unsigned long long>(batchDivisor));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        td.batchStride = static_cast<uint64_t>(batchStride);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// DeriveCheckShape — 校验表 #5（dtype）/#6（批量参数）+ 维度/方向/扫描几何派生。
// 派生口径与 prune_host 逐字段对齐（dtype/维度映射/pruneAlongRow）。
// ----------------------------------------------------------------------------
aclsparseStatus_t DeriveCheckShape(const aclsparseLtMatmulDescriptor* md, bool sparseA,
    PruneCheckParams& params)
{
    auto& td = params.tiling;
    const auto* sparse = sparseA ? md->matA : md->matB;
    // #5：dtype 白名单（先于 #6 批量校验：batchStride 对齐约束的除数依赖
    // dataBytes，且跨返回码可观测的顺序为 dtype NOT_SUPPORTED 优先）。
    // 分组大小由 kernel 按 sizeof(U) 编译期推导（FP32→g=2，其余→g=4），
    // 不落入 tiling data。
    switch (sparse->valueType) {
        case ACL_FLOAT:
            td.dataBytes = 4;
            break;
        case ACL_FLOAT16:
        case ACL_BF16:
            td.dataBytes = 2;
            break;
        case ACL_INT8:
            td.dataBytes = 1;
            break;
        default:
            OP_LOGE(kLogTag, "unsupported dtype (only FP32/FP16/BF16/INT8)");
            return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // #6：批量参数（DeriveBatchParams 承载，dtype 先行）。
    aclsparseStatus_t status = DeriveBatchParams(sparse, td.dataBytes, params);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // 维度映射（prune 同款）：A-sparse: (m,k)=(md_m,md_k)；
    // B-sparse: (m,k)=(md_k,md_n)（B rows = 逻辑 k，B cols = 逻辑 n）。
    const int64_t m = sparseA ? md_m(md) : md_k(md);
    const int64_t k = sparseA ? md_k(md) : md_n(md);
    // 剪枝方向（与 prune 完全一致）：沿行分组条件 = trans != isRowOrder。
    const bool sparseTrans = sparseA ? md_transA(md) : md_transB(md);
    const bool isRowOrder = sparse->order == ACL_SPARSE_ORDER_ROW;
    const bool alongRowPrune = (sparseTrans != isRowOrder);
    // 扫描几何（d_in 一律为行主序物理存储、行步长 ld；order 仅决定校验方向）：
    //   alongRowPrune=1（N+ROW / T+COL）→ RowScan：R=m × C=k，行内连续分组；
    //   alongRowPrune=0 且 trans（T+ROW）→ RowScan：R=k × C=m，行内连续分组；
    //   alongRowPrune=0 且非 trans（N+COL）→ ColScan：R=m × C=k，同列跨行分组。
    if (alongRowPrune) {
        td.rows = static_cast<uint64_t>(m);
        td.cols = static_cast<uint64_t>(k);
        td.alongRow = true;
    } else if (sparseTrans) {
        td.rows = static_cast<uint64_t>(k);
        td.cols = static_cast<uint64_t>(m);
        td.alongRow = true;
    } else {
        td.rows = static_cast<uint64_t>(m);
        td.cols = static_cast<uint64_t>(k);
        td.alongRow = false;
    }
    td.ld = static_cast<uint64_t>(sparse->ld);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// ValidateScanSpan — 校验表 #11：全量扫描跨度（元素 → 字节 → 指针加法）溢出。
// 总跨度 = (numBatches−1)×batchStride + (rows−1)×ld + cols（numBatches==1 时批量
// 项为 0，与扩展前公式一致）；CheckedMulAdd 逐项防 uint64 乘加回绕。
// 按命中阶段给出具体原因与上限：单矩阵跨度 / 叠加批量跨度 / 字节换算以
// INT64_MAX（kAddressLimit）为上限，末段指针加法以 uintptr 地址空间为上限。
// ----------------------------------------------------------------------------
aclsparseStatus_t ValidateScanSpan(const PruneCheckTilingData& td, uintptr_t dInAddr)
{
    uint64_t spanElems = 0;
    uint64_t totalElems = 0;
    uint64_t spanBytes = 0;
    const char* overflowItem = nullptr;
    uint64_t overflowLimit = 0;
    if (!CheckedMulAdd(td.rows - 1, td.ld, td.cols, spanElems)) {
        overflowItem = "matrix span elements (rows-1)*ld+cols";
        overflowLimit = kAddressLimit;
    } else if (!CheckedMulAdd(td.numBatches - 1, td.batchStride, spanElems, totalElems)) {
        overflowItem = "total span elements (numBatches-1)*batchStride+(rows-1)*ld+cols";
        overflowLimit = kAddressLimit;
    } else if (!CheckedMulAdd(totalElems, td.dataBytes, 0, spanBytes)) {
        overflowItem = "total span bytes ((numBatches-1)*batchStride+(rows-1)*ld+cols)*dataBytes";
        overflowLimit = kAddressLimit;
    } else if (spanBytes > std::numeric_limits<uintptr_t>::max() - dInAddr) {
        overflowItem = "input end address dIn+totalBytes";
        overflowLimit = std::numeric_limits<uintptr_t>::max();
    }
    if (overflowItem != nullptr) {
        OP_LOGE(kLogTag,
                "scan span overflow: %s exceeds max addressable %llu (rows=%llu ld=%llu cols=%llu "
                "numBatches=%llu batchStride=%llu dataBytes=%u)",
                overflowItem,
                static_cast<unsigned long long>(overflowLimit),
                static_cast<unsigned long long>(td.rows),
                static_cast<unsigned long long>(td.ld),
                static_cast<unsigned long long>(td.cols),
                static_cast<unsigned long long>(td.numBatches),
                static_cast<unsigned long long>(td.batchStride),
                td.dataBytes);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// ValidateCheckBuffers — 校验表 #7-#10：指针 / 对齐 / 维度 / ld 下限；
// #11 全量跨度（含 batch 项）溢出由 ValidateScanSpan 承载（原位调用）。
// ----------------------------------------------------------------------------
aclsparseStatus_t ValidateCheckBuffers(const void* dIn, int* dValid, PruneCheckParams& params)
{
    if (dIn == nullptr || dValid == nullptr) {
        OP_LOGE(kLogTag, "d_in/d_valid is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // #8：d_in 32 字节对齐（实现要求，严于官方 16B 惯例）：d_in 为 kernel
    // DataCopyPad 的 GM 源基地址，硬件要求 32B；16B 非 32B 对齐会进入未定义
    // 搬运路径（可能 507035 或读错数据）。rows/cols/ld/batchStride 均为
    // 32B/dataBytes 元素倍数 → 各 tile 源地址 = d_in + 32B 倍数，对齐性质
    // 由基址保证。
    if ((reinterpret_cast<uintptr_t>(dIn) % 32) != 0) {
        OP_LOGE(kLogTag, "d_in not 32-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto& td = params.tiling;
    // #9：扫描维度与 ld 的防御性复检（init 已保证对齐；上界统一复检 INT32_MAX（含 ld）——
    // 巨 ld+小 rows 可绕过 #11 跨度拦截致 kernel 越界读；ld≤INT32_MAX 则 srcStride<2^40）。
    const uint64_t divisor = static_cast<uint64_t>(kMatrixAlignmentBytes / td.dataBytes);
    const uint64_t int32Max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (td.rows == 0 || td.cols == 0 || td.ld == 0 || td.rows > int32Max || td.cols > int32Max ||
        td.ld > int32Max || td.rows % divisor != 0 || td.cols % divisor != 0 || td.ld % divisor != 0) {
        OP_LOGE(kLogTag,
                "invalid scan geometry: rows/cols/ld must be nonzero, <= %llu, and a multiple of "
                "%llu elements (32B/element size); got rows=%llu cols=%llu ld=%llu",
                static_cast<unsigned long long>(int32Max),
                static_cast<unsigned long long>(divisor),
                static_cast<unsigned long long>(td.rows),
                static_cast<unsigned long long>(td.cols),
                static_cast<unsigned long long>(td.ld));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // #10（D-2）：物理行长 = 扫描列数（RowScan 行长 C；ColScan 每物理行亦为 C 列，
    // 与方案 #10 公式 (pruneAlongRow==0 且 sparseTrans==1) ? m : k 等价）。
    // prune 在此场景静默 clamp 到 physRowLen（隐含紧致排布假设，可能越界读）；
    // 校验算子读错地址 = 错误结论 + 潜在 fault，故显式 INVALID_VALUE。
    if (td.ld < td.cols) {
        OP_LOGE(kLogTag, "ld=%llu < physical row length=%llu",
                static_cast<unsigned long long>(td.ld),
                static_cast<unsigned long long>(td.cols));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // #11：全量扫描跨度溢出（ValidateScanSpan 承载，按命中阶段给出原因与上限）。
    const uintptr_t dInAddr = reinterpret_cast<uintptr_t>(dIn);
    aclsparseStatus_t spanStatus = ValidateScanSpan(params.tiling, dInAddr);
    if (spanStatus != ACL_SPARSE_STATUS_SUCCESS) {
        return spanStatus;
    }
    params.input = dIn;
    params.dValid = dValid;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// validate_prune_check_params — 校验表 #1-#11 的顺序编排（顺序即优先级）。
// ----------------------------------------------------------------------------
aclsparseStatus_t validate_prune_check_params(aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr, const void* dIn, int* dValid,
    PruneCheckParams& params)
{
    // #1
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kLogTag, "handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // #2
    if (matmulDescr == nullptr || *matmulDescr == nullptr) {
        OP_LOGE(kLogTag, "matmulDescr is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // stream 可为 nullptr（ACL 默认流），不做非空校验。
    const auto* md = to_matmul_internal(*matmulDescr);
    const aclsparseLtMatDescriptor* sparse = nullptr;
    bool sparseA = false;
    // #3/#4（D-1 唯一稀疏侧）
    aclsparseStatus_t status = SelectSparseSide(md, sparse, sparseA);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // #5 dtype / #6 批量参数 + 维度/方向/扫描几何派生
    status = DeriveCheckShape(md, sparseA, params);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    // #7-#11
    return ValidateCheckBuffers(dIn, dValid, params);
}

// ----------------------------------------------------------------------------
// compute_prune_check_tiling — 构造 PruneCheckTilingData。
//   lanes = 256B/elemSize；tileCols = max(lanes, min(ceil(cols/lanes)×lanes, maxCols))；
//   maxCols = floor((min(GetUbSize(), 248KB) − 4096 − 256) / 16 / elem / lanes) × lanes；
//   inputPitch = tileCols × elem（lanes 倍数 → 恒 256B 对齐）；
//   tilesPerMatrix = ceil(rows/16) × tilesPerRowBand；totalTiles = tilesPerMatrix × numBatches；
//   blockDim = clamp(min(GetAivCoreCount(), totalTiles), 1, 64)。
// ----------------------------------------------------------------------------
aclsparseStatus_t compute_prune_check_tiling(PruneCheckParams& params)
{
    auto& td = params.tiling;
    const uint32_t coreCount = GetAivCoreCount();
    const uint64_t ubBytes = std::min(GetUbSize(), kUbLimit);
    if (coreCount == 0 || ubBytes <= kUbReserve + kReduceBufferBytes) {
        OP_LOGE(kLogTag, "insufficient platform resources: coreCount=%u ubBytes=%llu",
                coreCount, static_cast<unsigned long long>(ubBytes));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    const uint64_t lanes = kVectorBytes / td.dataBytes;
    const uint64_t budget = ubBytes - kUbReserve - kReduceBufferBytes;
    // dataBytes 已在 DeriveCheckShape 固定为 4/2/1（lanes 恒非零，无除零风险）。
    const uint64_t maxCols = budget / kTileRows / td.dataBytes / lanes * lanes;
    if (maxCols == 0) {
        OP_LOGE(kLogTag, "UB too small for one tile row (budget=%llu)",
                static_cast<unsigned long long>(budget));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    // 小 shape 场景按 ceil(cols/lanes)×lanes 收缩 tileCols，不强制扩满
    //（扫描型算子，buffer 大小只影响单次搬运粒度）。
    uint64_t tileCols = (td.cols + lanes - 1) / lanes * lanes;
    tileCols = std::max(lanes, std::min(tileCols, maxCols));
    // 防御性复检：dataBytes 由 DeriveCheckShape 固定为 4/2/1 →
    // lanes = 256B/dataBytes ≥ 64 → tileCols = max(lanes, …) ≥ 64，数学上恒
    // 非零（不可达路径）；显式校验使 tilesPerRowBand 的除法安全可证。
    if (tileCols == 0) {
        OP_LOGE(kLogTag, "tileCols unexpectedly zero (dataBytes=%u)", td.dataBytes);
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    td.tileRows = kTileRows;
    td.tileCols = static_cast<uint32_t>(tileCols);
    td.inputPitch = static_cast<uint32_t>(tileCols * td.dataBytes);
    td.inputBufferBytes = td.tileRows * td.inputPitch;
    td.tilesPerRowBand = (td.cols + tileCols - 1) / tileCols;
    const uint64_t rowBands = (td.rows + kTileRows - 1) / kTileRows;
    // totalTiles = 单 batch tile 数 × numBatches（batch 维度纳入 grid-stride
    // 领取，kernel 按 tileId / tilesPerMatrix 派生 batchIdx）。
    if (!CheckedMulAdd(rowBands, td.tilesPerRowBand, 0, td.tilesPerMatrix) ||
        !CheckedMulAdd(td.tilesPerMatrix, td.numBatches, 0, td.totalTiles)) {
        OP_LOGE(kLogTag, "totalTiles overflow: rowBands=%llu tilesPerRowBand=%llu numBatches=%llu",
                static_cast<unsigned long long>(rowBands),
                static_cast<unsigned long long>(td.tilesPerRowBand),
                static_cast<unsigned long long>(td.numBatches));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // grid-stride 天然均衡；totalTiles < 核数时部分核空转为已知限制（µs 级总耗时）。
    const uint64_t blockDim = std::min<uint64_t>(
        std::min<uint64_t>(coreCount, td.totalTiles), kMaxBlockDim);
    params.blockDim = static_cast<uint32_t>(std::max<uint64_t>(blockDim, 1));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// launch_prune_check_kernel — 校验表 #12：d_valid 预清零 + kernel 异步启动。
// d_valid 多 block 归约：memset 预清零 + 仅违规 block 写 1（多写者写同一值，
// 无 last-writer 歧义；memset 与 kernel 同 stream 保序，结果确定）。
// ----------------------------------------------------------------------------
aclsparseStatus_t launch_prune_check_kernel(const PruneCheckParams& params, aclrtStream stream)
{
    // 保留调用者的线程错误状态：已有错误会妨碍判断本次启动是否失败，
    // 因此在提交任务前返回失败（spmma_compress 同款探测）。
    const aclError previousError = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (previousError != ACL_SUCCESS) {
        OP_LOGE(kLogTag, "Pre-existing thread runtime error; prune-check not enqueued: %d",
                static_cast<int>(previousError));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    const aclError memsetError = aclrtMemsetAsync(params.dValid, sizeof(int), 0, sizeof(int), stream);
    if (memsetError != ACL_SUCCESS) {
        OP_LOGE(kLogTag, "aclrtMemsetAsync d_valid failed: %d", static_cast<int>(memsetError));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    OP_LOGD(kLogTag,
            "rows=%llu cols=%llu ld=%llu numBatches=%llu batchStride=%llu totalTiles=%llu "
            "dataBytes=%u alongRow=%d "
            "tileRows=%u tileCols=%u inputPitch=%u tilesPerRowBand=%u tilesPerMatrix=%llu",
            static_cast<unsigned long long>(params.tiling.rows),
            static_cast<unsigned long long>(params.tiling.cols),
            static_cast<unsigned long long>(params.tiling.ld),
            static_cast<unsigned long long>(params.tiling.numBatches),
            static_cast<unsigned long long>(params.tiling.batchStride),
            static_cast<unsigned long long>(params.tiling.totalTiles),
            params.tiling.dataBytes, static_cast<int>(params.tiling.alongRow),
            params.tiling.tileRows, params.tiling.tileCols, params.tiling.inputPitch,
            params.tiling.tilesPerRowBand,
            static_cast<unsigned long long>(params.tiling.tilesPerMatrix));
    OP_LOGI(kLogTag, "Launching prune-check with %u AIV blocks", params.blockDim);
    // 框架边界：kernel 启动 ABI（GM_ADDR = uint8_t*，<<<>>> 形参）无 const 形态，
    // 且 kernel 侧 GlobalTensor<const U> 不被 DataCopyPad 接受（见 kernel.cpp
    // inputGm_ 注释），const 链在此处终止——与 prune / spmma_compress host 的
    // 同款转换；kernel 全程只读 input，无下游误写路径。
    splt_prune_check_kernel_do(reinterpret_cast<GM_ADDR>(const_cast<void*>(params.input)),
        reinterpret_cast<GM_ADDR>(params.dValid), params.blockDim, params.tiling, stream);
    // 编译器生成的直接启动函数返回 void，因此读取线程运行时错误判断即时失败。
    // 此处不清除错误，也不等待异步执行完成（异步语义，对齐 prune D4/D5）。
    const aclError launchError = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (launchError != ACL_SUCCESS) {
        OP_LOGE(kLogTag, "Runtime error observed after prune-check launch: %d",
                static_cast<int>(launchError));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}
} // namespace

// ============================================================================
// Public API entry point（对齐 cuSPARSELt cusparseLtSpMMAPruneCheck 逐参）。
// ============================================================================
extern "C" aclsparseStatus_t aclsparseLtSpMMAPruneCheck(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    aclrtStream stream)
{
    PruneCheckParams params;
    aclsparseStatus_t status = validate_prune_check_params(handle, matmulDescr, d_in, d_valid,
                                                           params);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    status = compute_prune_check_tiling(params);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        return status;
    }
    return launch_prune_check_kernel(params, stream);
}
