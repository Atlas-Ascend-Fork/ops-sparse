/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of conditions of the CANN Open Software License Agreement Version 2.0
 * ("the License");
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// ============================================================================
// sparse2dense_host.cpp — aclsparseSparseToDense arch22（Atlas A2/A3，DAV_2201）公开 API 层。
//
// 职责：aclsparse 描述符校验与视图映射，委托 op_host 内部执行协议
// （校验 → tiling → 发射）：
//   aclsparseSparseToDense_bufferSize → 描述符校验 + 形状一致性（matA/matB 描述符两轴
//                             全等，对齐上游 arch35 ValidateDenseDescriptorMeta 范式）
//                             → Sparse2DenseGetBufferSize（恒返回 0）
//   aclsparseSparseToDense → ⓪ 描述符校验 + 形状一致性（两入口共用）
//                          → ① CheckMeta（元数据静态校验；含未声明 alias 存储区间
//                             重叠防护——dense 范围与 values/primary/secondary 任一
//                             非空区间相交即拒，含索引数组与部分重叠）
//                          → ② ValidateF3Cached（数据校验链，数据级约束；
//                             内容指纹守卫命中，未命中回退全量校验；默认 poll 形态下
//                             清零 kernel 经 D2H 后置钩子随校验链入队，先于任何 event
//                             操作，完成探测以 pinned 盐位自旋替代 event 同步）
//                          → ③ FillTiling（TilingData + TilingKey 编码）
//                          → ④ 同一 stream 先清零 kernel、后 scatter kernel（<<<>>> 直调，
//                             tiling 按值传参）
// 全异步返回：调用方读取/复用/释放输出前自行同步 stream；描述符、
// workspace 与 Device 数组在 stream 完成前有效（接口契约）。
// 清零/scatter kernel 为非模板宏展开内核入口（op_kernel/arch22/sparse2dense_kernel.cpp 的
// GENERATE_S2D_*_KERNEL 展开点），tiling 经 <<<>>> 按值下发，无 tiling GM
// 缓冲 / H2D 上传 / 尾部同步。
// 与既有 arch35 实现并存（sparse/CMakeLists.txt 按架构目录互斥收集），公共接口
// 签名复用 include/cann_ops_sparse.h 既有声明，本文件零修改公开头。
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <mutex>
#include <vector>

#include "log/log.h"

#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "op_host/sparse2dense_host.h"

// 清零与 scatter kernel 入口声明（op_kernel/arch22/sparse2dense_kernel_entries.h，Host/Kernel
// 共用契约头）：定义在 op_kernel/arch22/sparse2dense_kernel.cpp 的
// GENERATE_S2D_*_KERNEL 宏展开点；Host 侧 <<<>>> 直调
// （kernel_operator.h 提供 GM_ADDR 的 __gm__ 限定）；tiling 按值传参。
#include "op_kernel/arch22/sparse2dense_kernel_entries.h"

namespace {

constexpr const char *kTag = "aclsparseSparseToDense";

// 数据校验链指纹内核 launch grid 上限（partials 缓冲按 2×64×8B 预留；Host 侧
// min(coreNum, 64) 截断；与文件尾记忆化层的缓冲与缓存键口径一致）
constexpr uint32_t S2D_F3_HASH_MAX_BLOCKS = 64U;

// ---------------------------------------------------------------------------
// 命中稳态路径的完成探测模式开关（poll 形态）：
//   默认（poll）：指纹 partials D2H 入队后，经 onD2hEnqueued 钩子将清零 kernel 在
//     任何 event 操作之前入队（不在 record↔sync 窗口内，非挂死形态）；完成探测
//     以 pinned 盐位自旋替代 SynchronizeEvent——消除 event 同步唤醒引入的设备
//     空窗（~70µs/调用量级）。超时回退
//     「event 录制（清零 kernel 之后）→ 同步」，回退窗口内零发射，安全形态不变。
//   S2D_K1_POLL=0：关闭 poll（回退顺序形态：校验链返回后顺序提交清零 kernel），
//     供时序问题排查与 A/B 对照。
// ---------------------------------------------------------------------------
inline bool S2DF3PollDisabled()
{
    static const bool off = []() -> bool {
        const char *env = std::getenv("S2D_K1_POLL");
        return (env != nullptr && env[0] == '0');
    }();
    return off;
}

// poll 自旋预算（µs，默认 5000；hash 内核大 nnz 用例执行数百 µs 量级，预算
// 覆盖后仍超时即走 event 同步回退——正确性兜底，不挂死）
inline uint32_t S2DF3PollBudgetUs()
{
    static const uint32_t us = []() -> uint32_t {
        const char *env = std::getenv("S2D_K1_POLL_US");
        if (env == nullptr) {
            return 5000U;
        }
        const long v = strtol(env, nullptr, 10);
        return (v > 0 && v <= 1000000L) ? static_cast<uint32_t>(v) : 5000U;
    }();
    return us;
}

inline struct aclsparseContext *ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext *>(handle);
}

inline const struct aclsparseSpMatDescr *ToConstMatInner(aclsparseConstSpMatDescr_t desc)
{
    return reinterpret_cast<const struct aclsparseSpMatDescr *>(desc);
}

inline struct aclsparseDnMatDescr *ToDnMatInner(aclsparseDnMatDescr_t desc)
{
    return reinterpret_cast<struct aclsparseDnMatDescr *>(desc);
}

// ---------------------------------------------------------------------------
// 描述符层校验（接口契约口径：值类型不支持返回 NOT_SUPPORTED，空 handle 返回
// HANDLE_IS_NULLPTR，其余参数错误统一 INVALID_VALUE；形状一致性——matA/matB 描述符
// 两轴全等——由本层 ValidateShapeConsistency 承接（内部视图 S2DSpMatView 不携带
// matA rows/cols，lenPrimary 仅间接锁 CSR 行/CSC 列）；rows/cols/ld/nnz 域、
// len 匹配/dtype 一致性/别名区间重叠等由内部 CheckMeta 承接，见
// op_host/sparse2dense_tiling.cpp）
// ---------------------------------------------------------------------------
aclsparseStatus_t ValidateSpMatDescriptor(const struct aclsparseSpMatDescr *matInner)
{
    if (matInner->format != ACL_SPARSE_FORMAT_CSR &&
        matInner->format != ACL_SPARSE_FORMAT_CSC &&
        matInner->format != ACL_SPARSE_FORMAT_COO) {
        OP_LOGE(kTag, "unsupported format %d (CSR/CSC/COO required)", static_cast<int>(matInner->format));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matInner->ptrType != ACL_SPARSE_INDEX_32I || matInner->IdxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE(kTag, "unsupported index type ptr=%d idx=%d (only ACL_SPARSE_INDEX_32I)",
            static_cast<int>(matInner->ptrType), static_cast<int>(matInner->IdxType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matInner->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matInner->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(kTag, "unsupported indexBase %d", static_cast<int>(matInner->baseType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const aclDataType valType = matInner->valueType;
    if (valType != ACL_INT8 && valType != ACL_FLOAT16 && valType != ACL_BF16 &&
        valType != ACL_FLOAT && valType != ACL_COMPLEX64) {
        OP_LOGE(kTag, "unsupported valueType %d (int8/float16/bfloat16/float32/complex64 required)",
            static_cast<int>(valType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 描述符级形状一致性（两公开入口共用，第十轮评审整改）：matA 与 matB 描述符两轴
// 全等，不等即 INVALID_VALUE。判定委托 op_host 纯比较函数
// Sparse2DenseCheckShapeConsistency（uint64/int64 两轴类型安全收敛比较）；本层仅
// 补日志（对齐 arch35 ValidateDenseDescriptorMeta 的 dimension mismatch 风格）。
// 前提：matA 已通过 ValidateSpMatDescriptor（format/index/base/valueType 合法）。
aclsparseStatus_t ValidateShapeConsistency(const struct aclsparseSpMatDescr *matInner,
    const struct aclsparseDnMatDescr *dnInner)
{
    if (Sparse2DenseCheckShapeConsistency(matInner->rows, matInner->cols, dnInner->rows,
            dnInner->cols) != S2D_STATUS_SUCCESS) {
        OP_LOGE(kTag, "dimension mismatch: matA(%llu x %llu) vs matB(%ld x %ld)",
            static_cast<unsigned long long>(matInner->rows),
            static_cast<unsigned long long>(matInner->cols), dnInner->rows, dnInner->cols);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 描述符 → 内部视图（S2D_* 枚举与 cann_ops_sparse.h 枚举同值同序，直接位模式
// 转换；nnz 超界不在本层拦截，由内部 CheckMeta 统一返回 INVALID_VALUE；rows/cols
// 形状一致性已由上方 ValidateShapeConsistency 承接，不进视图）
void BuildSpMatView(const struct aclsparseSpMatDescr *matInner, struct S2DSpMatView *view)
{
    view->format = static_cast<int32_t>(matInner->format);
    view->indexBase = static_cast<int32_t>(matInner->baseType);
    view->nnz = static_cast<int64_t>(matInner->nnz);
    view->lenPrimary = (matInner->format == ACL_SPARSE_FORMAT_CSR)
        ? static_cast<int64_t>(matInner->rows) + 1
        : (matInner->format == ACL_SPARSE_FORMAT_CSC)
            ? static_cast<int64_t>(matInner->cols) + 1
            : static_cast<int64_t>(matInner->nnz);
    view->values = matInner->values;
    view->primary = static_cast<const int32_t *>(matInner->ptrs);
    view->secondary = static_cast<const int32_t *>(matInner->idxs);
    view->valueType = static_cast<int32_t>(matInner->valueType);
}

void BuildDnMatView(const struct aclsparseDnMatDescr *dnInner, struct S2DDnMatView *view)
{
    view->rows = dnInner->rows;
    view->cols = dnInner->cols;
    view->ld = dnInner->ld;
    view->layout = static_cast<int32_t>(dnInner->order);
    view->valueType = static_cast<int32_t>(dnInner->valueType);
    view->dense = dnInner->values;
}

aclsparseStatus_t ToAclStatus(S2DStatus st)
{
    return (st == S2D_STATUS_SUCCESS) ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

// ---------------------------------------------------------------------------
// 清零/scatter kernel <<<>>> 分发（TilingKey V_KIND → 宏展开入口；tiling 按值下发；grid=0
// 守卫，grid 语义：清零 kernel=核数全量、scatter kernel=min(blockNum, 核数)）
// ---------------------------------------------------------------------------
S2DStatus LaunchZeroKernel(uint32_t vKind, uint32_t grid, aclrtStream stream, void *denseDev,
    const Sparse2DenseTilingData &tiling)
{
    if (grid == 0U) {
        grid = 1U;
    }
    switch (vKind) {
        case SPARSE2DENSE_VK_INT8:      // TK0：uint16 通道造零（一次覆盖 2×int8）
            sparse2dense_zero_kernel_int8<<<grid, nullptr, stream>>>(
                (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_FLOAT16:   // TK1：half 通道
            sparse2dense_zero_kernel_fp16<<<grid, nullptr, stream>>>(
                (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_BFLOAT16:  // TK2：bfloat16_t 通道
            sparse2dense_zero_kernel_bf16<<<grid, nullptr, stream>>>(
                (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_FLOAT32:   // TK3：float 通道
            sparse2dense_zero_kernel_fp32<<<grid, nullptr, stream>>>(
                (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_COMPLEX64: // TK4：uint32 通道（实/虚两分量 +0f）
            sparse2dense_zero_kernel_c64<<<grid, nullptr, stream>>>(
                (uint8_t *)denseDev, tiling);
            break;
        default:
            return S2D_STATUS_INVALID_VALUE;
    }
    return S2D_STATUS_SUCCESS;
}

S2DStatus LaunchScatterKernel(uint32_t vKind, uint32_t grid, aclrtStream stream, void *valDev, void *primDev,
    void *secDev, void *denseDev, const Sparse2DenseTilingData &tiling)
{
    if (grid == 0U) {
        grid = 1U;
    }
    switch (vKind) {
        case SPARSE2DENSE_VK_INT8:      // TK0：int8 通道（SetValue 打包 + PIPE_ALL）
            sparse2dense_scatter_kernel_int8<<<grid, nullptr, stream>>>(
                (uint8_t *)valDev, (uint8_t *)primDev, (uint8_t *)secDev, (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_FLOAT16:   // TK1：uint16 通道（2B）
            sparse2dense_scatter_kernel_fp16<<<grid, nullptr, stream>>>(
                (uint8_t *)valDev, (uint8_t *)primDev, (uint8_t *)secDev, (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_BFLOAT16:  // TK2：uint16 通道（2B）
            sparse2dense_scatter_kernel_bf16<<<grid, nullptr, stream>>>(
                (uint8_t *)valDev, (uint8_t *)primDev, (uint8_t *)secDev, (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_FLOAT32:   // TK3：uint32 通道
            sparse2dense_scatter_kernel_fp32<<<grid, nullptr, stream>>>(
                (uint8_t *)valDev, (uint8_t *)primDev, (uint8_t *)secDev, (uint8_t *)denseDev, tiling);
            break;
        case SPARSE2DENSE_VK_COMPLEX64: // TK4：uint32 通道 ×2 单元/元素
            sparse2dense_scatter_kernel_c64<<<grid, nullptr, stream>>>(
                (uint8_t *)valDev, (uint8_t *)primDev, (uint8_t *)secDev, (uint8_t *)denseDev, tiling);
            break;
        default:
            return S2D_STATUS_INVALID_VALUE;
    }
    return S2D_STATUS_SUCCESS;
}

}  // namespace

// ===========================================================================
// 执行前置链：① CheckMeta（元数据静态校验）→ 核数动态获取（禁止硬编码；清零/scatter kernel grid
// 与指纹内核 grid 共用同一来源）→ ③ FillTiling（TilingData + TilingKey 编码）。
// ② 数据校验链（内容指纹守卫）在执行入口单独调用，返回后顺序提交清零 kernel
//（见 aclsparseSparseToDense）。coreNum 出参供清零/scatter kernel grid 使用；
// tilingKey 供调试观测（分发经 tiling.valueKind 完成，见下）。
// ===========================================================================
namespace {

aclsparseStatus_t PrepareExec(const struct S2DSpMatView *viewA, const struct S2DDnMatView *viewB, int32_t alg,
    aclrtStream stream, uint32_t &coreNum, Sparse2DenseTilingData &tiling)
{
    // ① 元数据静态校验（含 alg 合法性与未声明 alias 防护）
    S2DStatus st = Sparse2DenseCheckMeta(viewA, viewB, alg);
    if (st != S2D_STATUS_SUCCESS) {
        return ToAclStatus(st);
    }
    const uint32_t coreNumLocal = GetAivCoreCount();
    if (coreNumLocal == 0U) {
        OP_LOGE(kTag, "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    coreNum = coreNumLocal;
    // ③ TilingData 计算 + TilingKey 编码（key 供调试观测，不参与下方分发）
    uint64_t tilingKey = 0;
    return ToAclStatus(Sparse2DenseFillTiling(viewA, viewB, &tiling, &tilingKey));
}

// 执行入口前置：描述符静态校验 + 形状一致性 + handle 绑定 stream 提取 + 内部视图构建（bufferSize
// 恒 0 协议下执行接口 buffer 可空，见调用方）。失败返回错误码；成功输出 stream 与视图
aclsparseStatus_t ValidateAndBuildViews(aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB, aclrtStream &stream, struct S2DSpMatView *viewA, struct S2DDnMatView *viewB)
{
    const struct aclsparseSpMatDescr *matInner = ToConstMatInner(matA);
    aclsparseStatus_t st = ValidateSpMatDescriptor(matInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // 描述符级形状一致性（第十轮评审整改：CSR 仅 lenPrimary==rows+1 间接锁行、CSC
    // 仅锁列、COO 两轴全不锁——合法 COO matA=2×3 配 matB=1×1 原可全链静默输出错误
    // 尺寸；本层两轴全等校验为唯一全轴闸口，对齐上游 arch35 范式）
    st = ValidateShapeConsistency(matInner, ToDnMatInner(matB));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    stream = ToInternalHandle(handle)->stream;
    if (stream == nullptr) {
        OP_LOGE(kTag, "stream is nullptr, please call aclsparseSetStream first");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    BuildSpMatView(matInner, viewA);
    BuildDnMatView(ToDnMatInner(matB), viewB);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ② 数据校验链 + 清零 kernel 提交（PrepareExec 之后的执行段，数据级约束：
// offsets 单调、坐标 base 换算在界内）。内容指纹守卫命中——设备指纹 + D2H +
// 完成探测后按「指针 + 规格 + 指纹」比对，同指针原地改写必致指纹变化 → 未命中回退
// 全量校验（含 D2H 与 Host 坐标校验并同步返回错误码）；S2D_F3_CACHE_OFF 开关见
// 本文件记忆化层。
// 清零 kernel 提交时序约束（arch22/CANN 9.1）：
//   禁止把清零 kernel 发射插在 aclrtRecordEvent 与 aclrtSynchronizeEvent 之间
//     （该 record↔sync 窗口内发射会令设备流队列停摆：event flag 永不置位 + 后续
//     任务永不调度）；
//   event 同步唤醒在 D2H 完成与清零 kernel 启动之间引入设备空窗（~70µs/调用量级）；
//   现行 poll 形态：清零 kernel 经 onD2hEnqueued 钩子在指纹 D2H 入队后、任何 event 操作
//     之前入队（不在 record↔sync 窗口内）；完成探测以 pinned 盐位自旋
//     替代 SynchronizeEvent，稳态 stream 序 hash → D2H → 清零 → scatter 零空窗。
// 清零 kernel 不读索引内容、与校验结论无关，合法输入经 CheckMeta alias 防护后 dense 与
// 索引不重叠，校验结论为错误时清零 kernel 可能已清零 matB（错误返回时输出内容不在接口
// 契约内）。返回 SUCCESS 时清零 kernel 已在 stream 上提交。
// poll 形态：清零 kernel 经 D2H 后置钩子入队（校验链内部、任何 event 操作之前），
// 钩子失败/未触发时由调用方在校验链返回后兜底提交（submitted 标志防双发）
struct S2DZeroLaunchCtx {
    uint32_t vKind = 0;
    uint32_t grid = 0;
    aclrtStream stream = nullptr;
    void *dense = nullptr;
    const Sparse2DenseTilingData *tiling = nullptr;
    bool submitted = false;
};

void SubmitZeroOnD2hEnqueued(void *ctx)
{
    auto *z = static_cast<S2DZeroLaunchCtx *>(ctx);
    if (LaunchZeroKernel(z->vKind, z->grid, z->stream, z->dense, *z->tiling) == S2D_STATUS_SUCCESS) {
        z->submitted = true;
    }
}

aclsparseStatus_t ValidateAndSubmitZero(const struct S2DSpMatView *viewA, const struct S2DDnMatView *viewB,
    aclrtStream stream, uint32_t coreNum, const Sparse2DenseTilingData &tiling)
{
    const uint32_t hashBlocks = (coreNum < S2D_F3_HASH_MAX_BLOCKS) ? coreNum : S2D_F3_HASH_MAX_BLOCKS;
    // 清零 kernel 提交（poll 形态，默认）：清零 kernel 经 onD2hEnqueued 钩子在指纹 D2H 入队后、
    // 任何 event 操作之前入队——不在 record↔sync 窗口内（该窗口内发射会令流队列
    // 停摆）；校验链内部以 pinned 盐位自旋
    // 探测替代 SynchronizeEvent（消除同步唤醒的设备空窗，见文件头 poll 开关注释）。
    // 清零 kernel 不读索引内容、与校验结论无关，合法输入经 CheckMeta alias 防护后 dense 与
    // 索引不重叠，校验结论为错误时清零 kernel 可能已清零 matB（错误返回时输出内容不在
    // 接口契约内）。钩子未触发（S2D_K1_POLL=0 / 缓存关闭 / 空输入短路 / 采集失败）
    // 时由下方兜底路径在校验链返回后提交（顺序形态，语义不变）。返回 SUCCESS 时
    // 清零 kernel 已在 stream 上提交。
    if (!S2DF3PollDisabled()) {
        S2DZeroLaunchCtx zeroCtx;
        zeroCtx.vKind = tiling.valueKind;
        zeroCtx.grid = coreNum;
        zeroCtx.stream = stream;
        zeroCtx.dense = viewB->dense;
        zeroCtx.tiling = &tiling;
        const S2DStatus stF3 = Sparse2DenseValidateF3Cached(stream, hashBlocks, viewA, viewB,
            SubmitZeroOnD2hEnqueued, &zeroCtx);
        if (stF3 != S2D_STATUS_SUCCESS) {
            return ToAclStatus(stF3);
        }
        if (!zeroCtx.submitted) {
            const S2DStatus stZero = LaunchZeroKernel(tiling.valueKind, coreNum, stream, viewB->dense, tiling);
            if (stZero != S2D_STATUS_SUCCESS) {
                return ToAclStatus(stZero);
            }
        }
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const S2DStatus stF3 = Sparse2DenseValidateF3Cached(stream, hashBlocks, viewA, viewB);
    if (stF3 != S2D_STATUS_SUCCESS) {
        return ToAclStatus(stF3);
    }
    const S2DStatus stZero = LaunchZeroKernel(tiling.valueKind, coreNum, stream, viewB->dense, tiling);
    if (stZero != S2D_STATUS_SUCCESS) {
        return ToAclStatus(stZero);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

}  // namespace

// ===========================================================================
// Public APIs（签名复用 include/cann_ops_sparse.h 既有声明）
// ===========================================================================
extern "C" {

aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr || matB == nullptr || bufferSize == nullptr) {
        OP_LOGE(kTag, "matA/matB/bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const struct aclsparseSpMatDescr *matInner = ToConstMatInner(matA);
    aclsparseStatus_t st = ValidateSpMatDescriptor(matInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // 描述符级形状一致性（与执行入口共用，见 ValidateShapeConsistency）——查询与执行
    // 两入口同口径拒绝形状不匹配
    st = ValidateShapeConsistency(matInner, ToDnMatInner(matB));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    struct S2DSpMatView viewA;
    struct S2DDnMatView viewB;
    BuildSpMatView(matInner, &viewA);
    BuildDnMatView(ToDnMatInner(matB), &viewB);
    // 恒 0 协议：清零与 scatter 直写 matB，无 Device 临时副本，无 workspace
    return ToAclStatus(Sparse2DenseGetBufferSize(&viewA, &viewB, static_cast<int32_t>(alg), bufferSize));
}

aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matA == nullptr || matB == nullptr) {
        OP_LOGE(kTag, "matA or matB is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // bufferSize 恒 0 协议下执行接口 buffer 可空（查询为 0；本算子固有需求恒 0）
    (void)buffer;

    aclrtStream stream = nullptr;
    struct S2DSpMatView viewA;
    struct S2DDnMatView viewB;
    aclsparseStatus_t stViews = ValidateAndBuildViews(handle, matA, matB, stream, &viewA, &viewB);
    if (stViews != ACL_SPARSE_STATUS_SUCCESS) {
        return stViews;
    }

    // ①③ 执行前置链（CheckMeta → 核数 → FillTiling；见 PrepareExec）
    uint32_t coreNum = 0;
    Sparse2DenseTilingData tiling;
    aclsparseStatus_t stPre = PrepareExec(&viewA, &viewB, static_cast<int32_t>(alg), stream, coreNum, tiling);
    if (stPre != ACL_SPARSE_STATUS_SUCCESS) {
        return stPre;
    }

    // ② 数据校验链（内容指纹守卫）+ 清零 kernel 提交（顺序口径见
    //    ValidateAndSubmitZero）；返回 SUCCESS 时清零 kernel 已在 stream 上提交
    aclsparseStatus_t stVal = ValidateAndSubmitZero(&viewA, &viewB, stream, coreNum, tiling);
    if (stVal != ACL_SPARSE_STATUS_SUCCESS) {
        return stVal;
    }
    const uint32_t vKind = tiling.valueKind;

    // ④ scatter kernel 发射（清零 kernel 已在上方提交）：同一 stream 先清零后 scatter（覆写必须
    //    发生在清零完成后，保证输出全定义语义）；tiling 按值随 <<<>>> 下发（栈上
    //    结构体，无 GM 缓冲 / H2D）；scatter kernel grid = min(blockNum, coreNum)。发射后全异步
    //    返回（调用方读取/复用/释放输出前自行同步 stream）
    // nnz==0 快路径：输出契约是"非零元位置写入值、其余位置清零"，nnz==0 时输出
    // 必须为全零矩阵——不能 no-op 提前返回（校验链虽过，但会跳过清零、留下未
    // 初始化内存）。清零 kernel 完成后契约已满足，跳过 scatter kernel 发射（省一次
    // launch 开销），直接返回 SUCCESS
    if (viewA.nnz == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const uint32_t gridK2 = (tiling.blockNum == 0U || tiling.blockNum > coreNum) ? coreNum : tiling.blockNum;
    return ToAclStatus(LaunchScatterKernel(vKind, gridK2, stream,
        const_cast<void *>(viewA.values), const_cast<int32_t *>(viewA.primary),
        const_cast<int32_t *>(viewA.secondary), viewB.dense, tiling));
}

}  // extern "C"

// ============================================================================
// 数据校验链记忆化层——内容指纹缓存。
//
// 全量校验链（pinned 申请 + D2H + stream 同步 + Host O(nnz) 坐标校验）开销大，
// 本层对其结果做内容指纹记忆化：设备侧指纹内核对 primary/secondary 全量元素采集
// FNV-1a 指纹（64-lane 向量化，O(nnz) 设备并行、带宽受限），按「指针 + 规格 +
// 指纹 + numBlocks」比对进程内缓存：
//   命中（同指针同规格同内容）→ 直接返回缓存结果（免 MB 级 D2H 与 Host 坐标校验）；
//   未命中（新输入或内容变化）→ 回退 Sparse2DenseValidateF3 全量链（语义不变）并登记。
// 命中以内容指纹为前提：调用方原地改写索引数组时指针与规格
// 不变，仅按指针缓存会复用陈旧 SUCCESS，使新的越界坐标直接进入 scatter 越界写
// matB——任何原地内容变更必然改变指纹 → 未命中 → 全量重校，该路径被指纹封死。
// 资源（指纹 Device 缓冲、pinned 回读缓冲、event）与条目表均按 devId 隔离常驻
//（g_devCache 表键控）：同进程多卡切换各持独立状态与键域，杜绝跨卡复用资源或校验
// 结论；任何初始化/启动失败均无感回退全量校验（校验语义不降级）。发射无返回值的
// 静默失败由一次性 nonce 盐兜底（陈旧 partials 盐位失配 → miss → 全量重校）。
// onHashEnqueued 回调钩子（poll 形态，语义 = 指纹 D2H 入队后、任何 event 操作
// 之前触发）：本文件调用方经它提交清零 kernel（见
// ValidateAndSubmitZero 注释）；非 poll 模式不触发，清零 kernel 由调用方兜底提交。
// 钩子触发点严格保持在任何 event 操作之前（arch22/CANN 9.1 下 record↔sync 窗口
// 内发射会令流队列停摆）。
// 已知限制：指纹仅作缓存守卫，同内容同指纹（2^-64 碰撞漏检风险）；
// 多 host 线程并发变更同一输入的窗口不在守卫范围（单线程调用场景）。
// ============================================================================

namespace {

// S2D_F3_HASH_MAX_BLOCKS（launch grid 上限）见文件头常量定义（与本层缓冲与缓存键共用）
constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
constexpr size_t PARTIALS_BYTES = 3UL * S2D_F3_HASH_MAX_BLOCKS * sizeof(uint64_t);

// 缓存条目：键 = 数据指针身份 + 规格 + 内容指纹；值 = 校验结果
struct S2DF3CacheEntry {
    const void *primary = nullptr;
    const void *secondary = nullptr;
    uint64_t lenPrimary = 0;
    uint64_t nnz = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int32_t format = 0;
    int32_t indexBase = 0;
    uint32_t numBlocks = 0;
    uint64_t hashP = 0;
    uint64_t hashS = 0;
    S2DStatus status = S2D_STATUS_INVALID_VALUE;
};

// 单卡常驻状态：Device 资源 + pinned 回读缓冲 + event + 该卡专属条目表。
// devId 为资源持有卡备忘（表键即 devId，条目键域随表隔离——跨卡指针不复用）。
struct S2DF3CacheState {
    bool ready = false;
    int32_t devId = -1;
    void *partialsDev = nullptr;   // 3×MAX×8B 设备指纹缓冲（[3b]=P、[3b+1]=S、[3b+2]=盐）
    void *partialsHost = nullptr;  // pinned 回读缓冲
    aclrtEvent event = nullptr;    // D2H 完成事件（复用）
    uint64_t nonceCounter = 0;     // 一次性盐计数器（严格递增，陈旧 partials 检出）
    uint32_t nextSlot = 0;         // 环形替换游标
    std::vector<S2DF3CacheEntry> entries;
};

// per-device 状态表（进程内常驻，互斥保护；单线程/串行调用场景）：
// 首卡初始化后切卡不共享 ready——新卡按需自建资源与条目表；回访旧卡命中其既有
// 状态。选择「按卡隔离」而非「切卡销毁重建」：跨卡释放需回持有卡上下文，且每卡
// 常驻成本仅 KB 级；条目表随状态结构体走，缓存键域即按卡隔离。
std::map<int32_t, S2DF3CacheState> g_devCache;
std::mutex g_cacheMu;

// 按块序折叠部分指纹（与设备侧分块划分、折叠次序绑定 → 同内容同指纹）
uint64_t FoldPartials(const uint64_t *partials, uint32_t numBlocks, uint32_t stride)
{
    uint64_t hash = FNV_OFFSET_BASIS;
    for (uint32_t b = 0; b < numBlocks; b++) {
        hash ^= partials[static_cast<size_t>(b) * stride];
        hash *= FNV_PRIME;
    }
    return hash;
}

// 半途初始化失败：释放已分配资源并全字段复位（此时仍处于持有卡上下文，释放语义
// 正确；下次调用从零重试，无泄漏）。常驻资源仅在此失败路径释放，正常生命周期随进程。
void ReleaseCacheResources(S2DF3CacheState &cache)
{
    if (cache.event != nullptr) {
        (void)aclrtDestroyEvent(cache.event);
    }
    if (cache.partialsHost != nullptr) {
        (void)aclrtFreeHost(cache.partialsHost);
    }
    if (cache.partialsDev != nullptr) {
        (void)aclrtFree(cache.partialsDev);
    }
    cache = S2DF3CacheState{};
}

// 在当前 device 上初始化该卡的常驻资源（资源随 S2DF3CacheState 归属卡隔离）
bool EnsureCacheState(S2DF3CacheState &cache)
{
    if (cache.ready) {
        return true;
    }
    int32_t devId = -1;
    if (aclrtGetDevice(&devId) != 0) {
        return false;
    }
    if (aclrtMalloc(&cache.partialsDev, PARTIALS_BYTES, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        ReleaseCacheResources(cache);
        return false;
    }
    if (aclrtMallocHost(&cache.partialsHost, PARTIALS_BYTES) != 0) {
        ReleaseCacheResources(cache);
        return false;
    }
    if (aclrtCreateEvent(&cache.event) != 0) {
        ReleaseCacheResources(cache);
        return false;
    }
    cache.devId = devId;
    cache.entries.reserve(S2D_F3_HASH_MAX_BLOCKS);
    cache.ready = true;
    return true;
}

bool EntryMatches(const S2DF3CacheEntry &e, const S2DSpMatView *matA, const S2DDnMatView *matB, uint32_t numBlocks,
    uint64_t hashP, uint64_t hashS)
{
    return e.primary == matA->primary && e.secondary == matA->secondary && e.lenPrimary == matA->lenPrimary &&
        e.nnz == matA->nnz && e.rows == matB->rows && e.cols == matB->cols && e.format == matA->format &&
        e.indexBase == matA->indexBase && e.numBlocks == numBlocks && e.hashP == hashP && e.hashS == hashS;
}

// ---------------------------------------------------------------------------
// 诊断开关（首调用读取一次进程内缓存；C++11 magic static 保证多线程初始化安全，
// 免稳态路径每调用重复 getenv）：S2D_F3_CACHE_OFF=1 绕过记忆化（全量校验）、
// S2D_F3_HASH_GRID 覆盖指纹 grid
//（仅进程启动前设置生效——开关语义为运行口径选择，非运行中切换）
// ---------------------------------------------------------------------------
bool F3CacheDisabled()
{
    static const bool off = std::getenv("S2D_F3_CACHE_OFF") != nullptr;
    return off;
}

// S2D_F3_HASH_GRID 覆盖值（0 = 无覆盖；>0 且 ≤ 上限才生效）
uint32_t F3HashGridOverride()
{
    static const uint32_t grid = []() -> uint32_t {
        const char *env = std::getenv("S2D_F3_HASH_GRID");
        if (env == nullptr) {
            return 0U;
        }
        const long value = strtol(env, nullptr, 10);
        return (value > 0 && value <= static_cast<long>(S2D_F3_HASH_MAX_BLOCKS)) ? static_cast<uint32_t>(value)
                                                                                 : 0U;
    }();
    return grid;
}

// ---------------------------------------------------------------------------
// 指纹采集两段式：SubmitHashes 同 stream 顺序入队指纹内核 + partials D2H
//（poll 模式下随后触发 onD2hEnqueued 钩子——清零 kernel 在此入队，先于任何 event 操作，
// 不在 record↔sync 窗口内；poll 模式不录制 event）；AwaitHashes 在 poll 模式下
// 以 pinned 盐位自旋探测 D2H 完成（替代 SynchronizeEvent 的同步唤醒，消除
// 设备空窗），超时回退「event 录制（清零 kernel 之后）→ 同步」——回退窗口内零发射，
// 与挂死形态（record↔sync 窗口内发射）结构不同。任一环节失败（含 nonce 盐位
// 失配）按采集失败处理（调用方无感回退全量校验，校验语义不降级）。
// ---------------------------------------------------------------------------
bool SubmitHashes(void *stream, uint32_t numBlocks, const S2DSpMatView *matA, S2DF3CacheState &cache,
    uint64_t &nonce, bool pollMode, S2DF3OnHashEnqueued onD2hEnqueued, void *onD2hContext)
{
    if (!EnsureCacheState(cache)) {
        return false;
    }
    // 一次性盐：严格递增不重复；内核将其随块指纹写回（partials[3b+2]），AwaitHashes
    // 逐块比对——发射静默失败残留的陈旧 partials 盐位必失配，转为 miss → 全量校验
    nonce = ++cache.nonceCounter;
    const S2DF3HashParams params{static_cast<uint64_t>(matA->lenPrimary), static_cast<uint64_t>(matA->nnz), nonce};
    // 空数组传非空占位指针（指纹内核按长度守卫不访存）
    void *primaryGm = (matA->lenPrimary > 0) ? const_cast<int32_t *>(matA->primary) : cache.partialsDev;
    void *secondaryGm = (matA->nnz > 0) ? const_cast<int32_t *>(matA->secondary) : cache.partialsDev;
    // <<<>>> 发射参数须为简单表达式，stream 在此先行定型；发射为异步语句无返回值
    //，失败观测依赖同 stream 后置链：partials D2H +
    // 完成探测（poll 盐位 / event 同步）任一失败均回退全量校验；发射静默失败由
    // nonce 盐位失配兜底
    const aclrtStream stream_ = static_cast<aclrtStream>(stream);
    sparse2dense_f3_hash_kernel<<<numBlocks, nullptr, stream_>>>(
        (uint8_t *)primaryGm, (uint8_t *)secondaryGm, params, (uint8_t *)cache.partialsDev);
    const size_t partialsBytes = 3UL * numBlocks * sizeof(uint64_t);
    if (aclrtMemcpyAsync(cache.partialsHost, partialsBytes, cache.partialsDev, partialsBytes,
            ACL_MEMCPY_DEVICE_TO_HOST, static_cast<aclrtStream>(stream)) != 0) {
        return false;
    }
    if (pollMode) {
        // 清零 kernel（或等价发射）在 D2H 之后、任何 event 操作之前入队——稳态路径不录制
        // event，完成探测由 AwaitHashes 的盐位自旋承接
        if (onD2hEnqueued != nullptr) {
            onD2hEnqueued(onD2hContext);
        }
        return true;
    }
    if (aclrtRecordEvent(cache.event, static_cast<aclrtStream>(stream)) != 0) {
        return false;
    }
    return true;
}

// 单趟盐位扫描：全部块 nonce 匹配返回 true（自旋探测与同步后核验共用；
// volatile 读 pinned 内存——自旋防编译器优化成单次读取，同步后读取值等价）
bool ScanSaltsMatch(const volatile uint64_t *partials, uint32_t numBlocks, uint64_t nonce)
{
    for (uint32_t b = 0; b < numBlocks; b++) {
        if (partials[3UL * b + 2UL] != nonce) {
            return false;
        }
    }
    return true;
}

// event 同步 + 盐位核验 + 折叠（poll 超时回退与非 poll 稳态共用）：同步等待
// stream 尾部（含指纹内核与 partials D2H）落定后，先核盐位再按块序折叠。
// 任一失配返回 false → 调用方回退全量校验（校验语义不降级）
bool SyncEventVerifyAndFold(S2DF3CacheState &cache, uint32_t numBlocks, uint64_t nonce,
    uint64_t &hashP, uint64_t &hashS)
{
    if (aclrtSynchronizeEvent(cache.event) != 0) {
        return false;
    }
    const auto *partials = static_cast<const uint64_t *>(cache.partialsHost);
    if (!ScanSaltsMatch(partials, numBlocks, nonce)) {
        return false;
    }
    hashP = FoldPartials(partials, numBlocks, 3U);
    hashS = FoldPartials(partials + 1, numBlocks, 3U);
    return true;
}

// poll 形态：pinned 盐位自旋探测 + 撕裂二次复核 + 超时回退（完成探测语义见
// AwaitHashes 总注释）。返回 false 即采集失败 → 调用方回退全量校验
bool AwaitHashesPoll(S2DF3CacheState &cache, void *stream, uint32_t numBlocks, uint64_t nonce,
    uint64_t &hashP, uint64_t &hashS)
{
    // pinned 盐位自旋探测（volatile 读；D2H 完成即命中，hash 内核执行期间空转
    // 等待——与 SynchronizeEvent 的睡眠唤醒不同，无 ~数十 µs 唤醒空窗）
    auto *partialsV = static_cast<volatile const uint64_t *>(cache.partialsHost);
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::microseconds(S2DF3PollBudgetUs());
    bool seen = ScanSaltsMatch(partialsV, numBlocks, nonce);
    while (!seen && std::chrono::steady_clock::now() < deadline) {
        seen = ScanSaltsMatch(partialsV, numBlocks, nonce);
    }
    if (!seen) {
        // 超时回退：清零 kernel 已入队于本 event 之前——录制→同步窗口内零发射（安全
        // 形态，与挂死形态结构不同）；同步后按盐位再核（D2H 迟到完成亦可命中）
        if (aclrtRecordEvent(cache.event, static_cast<aclrtStream>(stream)) != 0) {
            return false;
        }
        return SyncEventVerifyAndFold(cache, numBlocks, nonce, hashP, hashS);
    }
    // 命中：折叠后盐位二次复核（撕裂读防护；单线程下盐位严格递增，二次失配
    // 仅在异常重排时出现——保守转失败走全量校验）
    const auto *partials = static_cast<const uint64_t *>(cache.partialsHost);
    hashP = FoldPartials(partials, numBlocks, 3U);
    hashS = FoldPartials(partials + 1, numBlocks, 3U);
    return ScanSaltsMatch(partialsV, numBlocks, nonce);
}

// 非 poll 形态（S2D_K1_POLL=0 回退顺序提交行为）：event 已在 SubmitHashes 录制，
// 直接同步→核验→折叠
bool AwaitHashesEvent(S2DF3CacheState &cache, uint32_t numBlocks, uint64_t nonce,
    uint64_t &hashP, uint64_t &hashS)
{
    return SyncEventVerifyAndFold(cache, numBlocks, nonce, hashP, hashS);
}

// 盐位逐块比对（陈旧 partials 检出，见 S2DF3HashParams）：任一失配即采集失败 →
// 调用方回退全量校验。通过后按块序折叠（布局 [3b]=P、[3b+1]=S，stride=3）。
// poll / 非 poll 两形态分函数承接（各自保持小而清晰，poll 形态内含超时回退）
bool AwaitHashes(S2DF3CacheState &cache, void *stream, uint32_t numBlocks, uint64_t nonce,
    uint64_t &hashP, uint64_t &hashS, bool pollMode)
{
    if (pollMode) {
        return AwaitHashesPoll(cache, stream, numBlocks, nonce, hashP, hashS);
    }
    return AwaitHashesEvent(cache, numBlocks, nonce, hashP, hashS);
}

// 结果登记（环形替换，容量 = 指纹 grid 上限；指纹字段为本次采集的有效值）
void RegisterEntry(S2DF3CacheState &cache, const S2DSpMatView *matA, const S2DDnMatView *matB, uint32_t numBlocks,
    uint64_t hashP, uint64_t hashS, S2DStatus status)
{
    S2DF3CacheEntry entry;
    entry.primary = matA->primary;
    entry.secondary = matA->secondary;
    entry.lenPrimary = static_cast<uint64_t>(matA->lenPrimary);
    entry.nnz = static_cast<uint64_t>(matA->nnz);
    entry.rows = matB->rows;
    entry.cols = matB->cols;
    entry.format = matA->format;
    entry.indexBase = matA->indexBase;
    entry.numBlocks = numBlocks;
    entry.hashP = hashP;
    entry.hashS = hashS;
    entry.status = status;
    if (cache.entries.size() < S2D_F3_HASH_MAX_BLOCKS) {
        cache.entries.push_back(entry);
    } else {
        cache.entries[cache.nextSlot] = entry;
        cache.nextSlot = (cache.nextSlot + 1U) % S2D_F3_HASH_MAX_BLOCKS;
    }
}

}  // namespace

extern "C" S2DStatus Sparse2DenseValidateF3Cached(void *stream, uint32_t numBlocks, const struct S2DSpMatView *matA,
    const struct S2DDnMatView *matB, S2DF3OnHashEnqueued onHashEnqueued, void *onHashContext)
{
    if (stream == nullptr || matA == nullptr || matB == nullptr || numBlocks == 0U) {
        return Sparse2DenseValidateF3(stream, matA, matB);
    }
    // 空内容短路（lenPrimary==0 且 nnz==0 ⇔ COO nnz==0）：两数组零字节、无内容可
    // 变，指针键无陈旧风险；全量校验对空输入本就是纯 Host 路径（无 D2H、无 pinned
    // 申请，空 vector 由 Host 侧守卫），直接执行即得结论，免指纹采集的每调用
    // 固定开销（发射 + D2H + 同步）
    if (matA->lenPrimary == 0 && matA->nnz == 0) {
        return Sparse2DenseValidateF3(stream, matA, matB);
    }
    std::lock_guard<std::mutex> lock(g_cacheMu);

    // 诊断开关（首调用读取一次，见 F3CacheDisabled 等两 helper）
    if (F3CacheDisabled()) {
        return Sparse2DenseValidateF3(stream, matA, matB);
    }
    // device 隔离入口：以当前 device 选卡级状态表（常驻资源与条目键域均按 devId
    // 隔离，跨卡不复用）。取不到当前 device（无设备上下文）→ 回退全量校验
    // 且不登记（后续在有效卡上下文的调用不会误命中来历不明的条目）
    int32_t devId = -1;
    if (aclrtGetDevice(&devId) != 0) {
        return Sparse2DenseValidateF3(stream, matA, matB);
    }
    S2DF3CacheState &cache = g_devCache[devId];

    const uint32_t gridOverride = F3HashGridOverride();
    if (gridOverride != 0U) {
        numBlocks = gridOverride;
    }
    if (numBlocks > S2D_F3_HASH_MAX_BLOCKS) {
        numBlocks = S2D_F3_HASH_MAX_BLOCKS;
    }

    // 内容指纹守卫（默认且唯一命中口径）：设备指纹内核对 primary/secondary 全量
    // 元素采集 FNV-1a 指纹（64-lane 向量化，O(nnz) 并行）+ partials D2H + 完成探测
    //（poll 盐位自旋 / event 同步回退），按「指针 + 规格 + 指纹 + numBlocks」比对
    // ——同指针原地改写内容必然改变指纹 → 未命中 → 全量重校，杜绝复用陈旧
    // SUCCESS 使越界坐标进入 scatter。采集任一环节失败无感回退全量校验（校验语义
    // 不降级，不登记缓存）。
    // onHashEnqueued 钩子（poll 形态，语义 = 指纹 D2H 入队后、任何 event 操作之前
    // 触发）：调用方经它提交清零 kernel；调用方 stream 顺序为 hash → D2H → 清零 → scatter。
    uint64_t nonce = 0;
    uint64_t hashP = 0;
    uint64_t hashS = 0;
    // poll 形态（默认）：onHashEnqueued 钩子（语义 = 指纹 D2H 入队后）在
    // SubmitHashes 内部、任何 event 操作之前触发——清零 kernel 在此入队，不在
    // record↔sync 窗口内（原挂死形态为 record 与 sync 之间触发，已移除）。
    // 非 poll 模式（S2D_K1_POLL=0）不触发钩子，清零 kernel 由调用方在校验链返回后提交
    const bool pollMode = !S2DF3PollDisabled() && onHashEnqueued != nullptr;
    if (SubmitHashes(stream, numBlocks, matA, cache, nonce, pollMode, onHashEnqueued, onHashContext)) {
        if (AwaitHashes(cache, stream, numBlocks, nonce, hashP, hashS, pollMode)) {
            for (const S2DF3CacheEntry &e : cache.entries) {
                if (EntryMatches(e, matA, matB, numBlocks, hashP, hashS)) {
                    return e.status;
                }
            }
            // 全量校验（含 D2H 与 Host 坐标校验，同步返回错误码），结果登记本卡环形缓存；
            // 此时清零 kernel 已入队（event 之后），全量校验的 D2H 排于其后——首调/内容变更
            // 路径的固有串行，稳态命中不受影响
            const S2DStatus status = Sparse2DenseValidateF3(stream, matA, matB);
            RegisterEntry(cache, matA, matB, numBlocks, hashP, hashS, status);
            return status;
        }
    }
    return Sparse2DenseValidateF3(stream, matA, matB);
}
