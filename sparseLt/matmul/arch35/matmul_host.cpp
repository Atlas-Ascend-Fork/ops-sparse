/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file matmul_host.cpp
 * \brief aclsparseLtMatmul host-side API implementation.
 * Launches the cube matmul + epilogue (splitK>1) or fused matmul+epilogue
 * (splitK==1) device kernels via extern "C" launchers declared in
 * matmul/arch35/matmul_kernel.h. Fills alpha/beta into the tiling POD and
 * pushes it to device workspace[tilingOff] before launching.
 */

// Include order: C standard → C++ → CANN → local.
#include <cstdint>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "log/log.h"
#include "cann_ops_sparseLt.h"
#include "aclsparselt_handle_internal.h"
#include "shared/aclsparselt_internal.h"
#include "matmul/arch35/matmul_kernel.h"

// ============================================================================
// Matmul — launch cube matmul + epilogue. Fills alpha/beta into tiling.
// aclsparseLtMatmul = validate_pointer_alignment + validate_matmul_params +
// prepare_matmul_pointers + launch_matmul_kernels. The validation logic,
// error codes, log messages and kernel launch order are unchanged. The
// public extern "C" signature is preserved.
// ============================================================================

// Bundled resolved state for the matmul kernel launch: produced by
// prepare_matmul_pointers and consumed by launch_matmul_kernels.
struct MatmulLaunchArgs {
    AclsparseltTilingData td;   // copy of baseTD with alpha/beta filled
    void* aPruned;
    void* bMat;
    void* temp;
    void* cMat;
    void* dMat;
    void* tilingGm;
    int32_t dt;
};

// validate_pointer_alignment — 16-byte alignment check for all data pointers
// (predicate shared via splt_data_ptrs_aligned16, MatmulSearch 校验同构).
static aclsparseStatus_t validate_pointer_alignment(const void* workspace,
                                                    const void* matA, const void* matB,
                                                    const void* matC, const void* matD)
{
    // Validate 16-byte alignment of data pointers (NVIDIA
    // requires all pointers aligned to 16 bytes).
    if (!splt_data_ptrs_aligned16(workspace, matA, matB, matC, matD)) {
        OP_LOGE(kSparseLtLogTag, "Matmul: data pointer not 16-byte aligned");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ============================================================================
// GM layout integration ((b), unified for both sides): cube transpose-flag
// flip for the COL transpose view. The inputs' physical row strides pass
// td.ldA/td.ldB (PlanInit snapshots, same caliber as ldC/ldD) straight into
// the row-stride parameters of the kernel GM layouts NDExt/DNExt; MTE2 moves
// skip row-tail padding row by row via the row stride — non-compact inputs
// with ld need no preprocessing at all.
// COL 描述符的物理缓冲 ≡ 其转置的 ROW 形态（元素 (i,j) 位于 i + j*ld，
// ld 即该转置视图的行步长），转置读由既有 DNExt 转置机制承载：翻转
// td.sparseTrans / td.transB 即可，零拷贝零 kernel 改动；ld 口径在
// ROW/COL 间不变（ldA/ldB 均直读描述符 ld）。
// (b) 两侧同规则：稀疏侧（A_pruned/B_pruned 为 prune 的描述符布局输出）
// 与 dense 侧（描述符布局输入）统一按 order==COL 翻转——最终标志 =
// isColView(该侧) = ((op==T) != (order==COL))，无论物理布局如何，投递到
// L0 的逻辑矩阵恒为 op(该侧)。
// ============================================================================
static aclsparseStatus_t splt_apply_gm_layout(const aclsparseLtMatmulDescriptor* md,
                                              AclsparseltTilingData& td)
{
    if (md->matA != nullptr && md->matA->order == ACL_SPARSE_ORDER_COL) {
        td.sparseTrans = 1 - td.sparseTrans;
    }
    if (md->matB != nullptr && md->matB->order == ACL_SPARSE_ORDER_COL) {
        td.transB = 1 - td.transB;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_matmul_params — all null/alignment/transpose checks. outMd/outBaseTD
// return the plan internals extracted for downstream prepare_matmul_pointers.
static aclsparseStatus_t validate_matmul_params(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    aclrtStream* streams, int32_t numStreams,
    const void* workspace,
    const void* matA, const void* matB, const void* matC, const void* matD,
    aclsparseLtMatmulDescriptor*& outMd,
    AclsparseltTilingData*& outBaseTD,
    aclrtStream& outStream)
{
    // Return HANDLE_IS_NULLPTR for null handle, consistent
    // with all other APIs (was INVALID_VALUE before).
    if (handle == nullptr || *handle == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: handle is null");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr || *plan == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: plan is null");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Validate plan internal pointers — a partially
    // initialized or corrupted plan would otherwise cause null dereference.
    auto* md = (*plan)->matmulDescr;
    auto* baseTd = (*plan)->tilingData;
    if (md == nullptr || baseTd == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: plan internals null (matmulDescr/tilingData)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Validate streams: streams == nullptr means "default stream only"
    // (lenient policy, aligned with the cuSPARSELt official samples that
    // declare `int num_streams = 0; cudaStream_t* streams = nullptr;` and
    // pass them directly). numStreams is not judged in that case — the
    // count has no consumer when the array is absent, symmetric with the
    // existing (non-null, numStreams > 1) tolerance that only uses
    // streams[0]. When streams != nullptr, numStreams must be > 0.
    // streams[0] itself may be nullptr — the ACL runtime treats 0/nullptr
    // as the default stream, which is valid for all aclrtMemcpy /
    // kernel-launch APIs.
    if (streams == nullptr) {
        outStream = nullptr;  // ACL default stream
    } else {
        if (numStreams <= 0) {
            OP_LOGE(kSparseLtLogTag, "Matmul: numStreams<=0 with non-null streams");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        outStream = streams[0];
    }
    if (workspace == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: workspace is null");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    aclsparseStatus_t st = validate_pointer_alignment(workspace, matA, matB, matC, matD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // transA and transB are now supported end-to-end:
    //   - transA is handled by the prune kernel (reads physical (k,m) layout,
    //     outputs A_pruned as (m,k) row-major). The matmul kernel reads A_pruned
    //     unchanged (always (m,k)).
    //   - transB is passed to the matmul kernel via td.transB; the kernel
    //     declares GM B as (n,k) and transposes during CopyGM2L1 (NDExt→ZN).
    outMd = md;
    outBaseTD = baseTd;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// fill_matmul_tiling_fields — build a tiling copy with alpha/beta/bias/
// activation/batch config filled in from the matmul descriptor and caller
// arguments. The fill order and default values are preserved exactly.
// vec scaling 统一活读：alpha/beta 向量缩放模式执行期从描述符编码位
// 读取（decode_alpha_scaling / decode_beta_scaling，替代 baseTd 的 PlanInit
// 快照），与其余活读属性（bias/activation/batch）时序
// 语义一致——PlanInit 后 SetAttribute 开/关向量缩放即时生效（宽容语义）。
// 后开 vec + Device 向量正常走向量分支；后关 vec 回标量分支；
// "PlanInit 快照无 vec + 传入 Device 指针"的标量分支 host 解引用崩溃域
// 随之消除（validate 层同源活读，见 validate_vector_scaling_pointers）。
static void fill_matmul_tiling_fields(
    const aclsparseLtMatmulDescriptor* md,
    const AclsparseltTilingData* baseTd,
    const void* alpha, const void* beta,
    MatmulLaunchArgs& args)
{
    args.td = *baseTd;
    const bool alphaVec = decode_alpha_scaling(md);
    const bool betaVec = decode_beta_scaling(md);
    if (alphaVec) {
        args.td.alphaVectorScaling = 1;
        args.td.alphaDevPtr = reinterpret_cast<uint64_t>(alpha);
        args.td.alpha = 1.0f;
    } else {
        args.td.alphaVectorScaling = 0;
        args.td.alphaDevPtr = 0;
        args.td.alpha = (alpha != nullptr) ? *static_cast<const float*>(alpha) : 1.0f;
    }
    if (betaVec) {
        args.td.betaVectorScaling = 1;
        args.td.betaDevPtr = reinterpret_cast<uint64_t>(beta);
        args.td.beta = 0.0f;
    } else {
        args.td.betaVectorScaling = 0;
        args.td.betaDevPtr = 0;
        args.td.beta = (beta != nullptr) ? *static_cast<const float*>(beta) : 0.0f;
    }
    args.dt = baseTd->dataType;

    // 从 matmul descriptor 读取 bias/activation 配置填入 tiling copy
    // （仿 alpha/beta 填充模式，覆盖 fill_tiling_dims 中的默认值）。
    args.td.biasDevPtr = (md->biasPointer != nullptr)
        ? reinterpret_cast<uint64_t>(md->biasPointer) : 0;
    args.td.biasStride = md->biasStride;
    // activationType 统一编码：0=无, 1=ReLU, 2=GeLU
    if (md->activationRelu != 0) {
        args.td.activationType = 1;
    } else if (md->activationGelu != 0) {
        args.td.activationType = 2;
    } else {
        args.td.activationType = 0;
    }
    args.td.reluUpperBound = md->reluUpperBound;
    args.td.reluThreshold = md->reluThreshold;
    args.td.geluScaling = md->geluScaling;
    // biasChunkMode 活读重算：该降级标志依赖 vec 状态
    // （compute_bias_chunk_mode 的 needVecScale 条件），vec 活读后 PlanInit
    // 时按快照计算的值可能过期——"PlanInit 无 vec → 后设 vec + bias 且 m
    // 超大"时全量加载路径 UB 超限（降级标志过期）。先归零再按活读后的
    // args.td 状态重算（与 PlanInit 同式；cube tiling 取 baseTd 固化值，
    // 与 kernel 实际消费的 baseM/baseN 一致）。正常时序（PlanInit 前设置
    // vec）下重算结果与 PlanInit 快照逐位一致，零回归。
    args.td.biasChunkMode = 0;
    const CubeTiling liveCt{baseTd->baseM, baseTd->baseN, baseTd->baseK};
    compute_bias_chunk_mode(args.td, md, args.dt, liveCt, md_m(md));
    // 从 mat descriptor 读取 batch 配置填入 tiling copy
    args.td.numBatches = (md->matA != nullptr && md->matA->numBatches > 0) ? md->matA->numBatches : 1;
    args.td.batchStrideA = (md->matA != nullptr) ? md->matA->batchStride : 0;
    args.td.batchStrideB = (md->matB != nullptr) ? md->matB->batchStride : 0;
    args.td.batchStrideC = (md->matC != nullptr) ? md->matC->batchStride : 0;
    args.td.batchStrideD = (md->matD != nullptr) ? md->matD->batchStride : 0;
}

// resolve_matmul_apruned — matA 解析（自
// resolve_matmul_pointers 的 if-else 链提取）。Honor matA per cuSPARSELt
// Matmul contract: when matA is non-null, read the (already pruned) A from
// matA; only fall back to workspace[aPrunedOffset] when matA is null. The
// fallback order, null checks, and error codes are preserved exactly
// (constraint: the null check after fallback is critical).
static aclsparseStatus_t resolve_matmul_apruned(
    const aclsparseLtMatmulDescriptor* md,
    const void* matA, void* workspace, MatmulLaunchArgs& args)
{
    if (matA == nullptr) {
        // dense×dense path (hasStructuredSparsity==false) — matA is the
        // original dense A and must be passed explicitly (no fallback).
        if (!md_hasStructuredSparsity(md)) {
            OP_LOGE(kSparseLtLogTag, "Matmul: matA is null (dense×dense requires explicit dense A)");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        // B-sparse: matA is the dense matrix and must be passed
        // explicitly; no workspace fallback (unlike A-sparse where A_pruned
        // can live in workspace[aPrunedOffset]).
        if (!md_isSparseA(md) || workspace == nullptr) {
            OP_LOGE(kSparseLtLogTag, "Matmul: matA is null (B-sparse requires explicit dense A; A-sparse requires workspace fallback)");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        // A-sparse 回退 × batch > 1 拒绝：workspace 的 aPruned 区为单 batch
        // 容量（(b) 稀疏侧物理跨度 span×elemSize），多 batch 时 kernel 按
        // batchStrideA 推进读取将越界——多 batch 场景须先显式调用
        // aclsparseLtSpMMAPrune 产出 A_pruned（batch 布局）再经 matA 传入。
        if (args.td.numBatches > 1) {
            OP_LOGE(kSparseLtLogTag,
                    "Matmul: matA is null with numBatches=%d (batch > 1 requires explicit A_pruned; "
                    "workspace aPruned region is single-batch)", args.td.numBatches);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        args.aPruned = static_cast<uint8_t*>(workspace) + args.td.aPrunedOffset;
    } else {
        args.aPruned = const_cast<void*>(matA);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// resolve_matmul_bmat — matB 解析（同上提取）：
// B/C/D must be passed explicitly to Matmul (aligned with cuSPARSELt). The
// null check after assignment is preserved exactly (checked by the caller
// resolve_matmul_pointers — the null check after fallback is critical).
static aclsparseStatus_t resolve_matmul_bmat(const void* matB, MatmulLaunchArgs& args)
{
    args.bMat = const_cast<void*>(matB);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// resolve_matmul_pointers — resolve aPruned (with matA fallback), bMat, cMat,
// dMat, temp, and tilingGm pointers. The fallback order, null checks, and
// error codes are preserved exactly (constraint: the null check after
// fallback is critical). matA/matB 的回退解析链见
// resolve_matmul_apruned / resolve_matmul_bmat。
static aclsparseStatus_t resolve_matmul_pointers(
    const aclsparseLtMatmulDescriptor* md,
    const void* matA, const void* matB, const void* matC, void* matD,
    void* workspace,
    MatmulLaunchArgs& args)
{
    aclsparseStatus_t st = resolve_matmul_apruned(md, matA, workspace, args);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = resolve_matmul_bmat(matB, args);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // temp only used when splitK > 1; fused path (splitK==1) bypasses temp.
    args.temp = (args.td.splitK > 1)
        ? static_cast<uint8_t*>(workspace) + args.td.tempResultOffset
        : nullptr;
    args.cMat = const_cast<void*>(matC);
    args.dMat = matD;
    if (args.bMat == nullptr || args.cMat == nullptr || args.dMat == nullptr) {
        OP_LOGE(kSparseLtLogTag, "Matmul: matB/matC/matD is null (must be passed explicitly)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    args.tilingGm = static_cast<uint8_t*>(workspace) + args.td.tilingOffset;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// prepare_matmul_pointers — tiling copy with alpha/beta fill, push to device,
// and pointer resolution (matA fallback, bMat/cMat/dMat fallback to descriptor
// values). The fallback order and the null check after fallback are preserved
// exactly (constraint: the null check after fallback is critical).
// validate_one_scaling_vector — 单个缩放向量（alpha/beta）的执行期校验
// （自 validate_vector_scaling_pointers 的 alpha/beta
// 对称段提取）。检查顺序与日志内容保持不变：
//   1. 16 字节对齐（对齐 cuSPARSELt "All pointers must be aligned to
//      16 bytes"）：缩放向量与 bias 同经 DataCopyPad 以 GM 源地址读取
//      （SpltLoadScalingUB 全量模式 / SpltFusedEpilogueGeneralPath 的
//      per-tile 段加载）——实测 DataCopyPad 容忍 16B 对齐 GM 源（16B
//      非 32B 基址探针全过，含 fullload 与 chunk 装载路径，回归锚点见
//      ltmatmul_test_align16.cpp；原 32B 口径为历史收紧，D5 轮实验证伪
//      后回摆 16B。chunk 段加载的源偏移 mPos*4 为 baseM*4 的倍数，恒
//      16B 对齐，基址对齐即充分）。
//   2. device 内存判定：经 aclrtPointerGetAttributes，非 Device 内存
//      提前拒绝（把不可诊断的设备 AI Core Error 转化为带日志的 host
//      侧报错）。
static aclsparseStatus_t validate_one_scaling_vector(const void* p, const char* name)
{
    if ((reinterpret_cast<uintptr_t>(p) % 16) != 0) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: %s vector pointer not 16-byte aligned (DataCopyPad GM "
                "source requirement)",
                name);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclrtPtrAttributes attr;
    aclError ret = aclrtPointerGetAttributes(p, &attr);
    if (ret != ACL_SUCCESS) {
        // 查询失败时 attr 未被写入，不打 attr.location.type（未初始化读，
        // 仅日志内容问题），只打返回码。
        OP_LOGE(kSparseLtLogTag,
                "Matmul: %s vector scaling enabled but aclrtPointerGetAttributes failed "
                "(ret=%d) — pass a device float[M] pointer",
                name, static_cast<int>(ret));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (attr.location.type != ACL_MEM_LOCATION_TYPE_DEVICE) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: %s vector scaling enabled but pointer is not device memory "
                "(locType=%d) — pass a device float[M] pointer",
                name, static_cast<int>(attr.location.type));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_vector_scaling_pointers — 向量缩放指针位置校验（防设备级崩溃）。
// 向量缩放启用时（含 BETA_VECTOR_SCALING 隐含提升 ALPHA 的组合），alpha/beta
// 将被 kernel 以 GM 地址直接读取（SpltLoadScalingUB 的 DataCopyPad 源）——
// 若调用方误传 host 指针（栈/堆），GM 侧 MTE 访问非法地址触发 AI Core
// Error（errcode 95，多核形态），且设备错误会级联（后续同流操作不可靠）。
// 本校验经 aclrtPointerGetAttributes 判定指针位置，非 Device 内存提前拒绝
// （INVALID_VALUE），把不可诊断的设备崩溃转化为带日志的 host 侧报错。
// vec 模式 nullptr 拦截（NVIDIA 契约对齐）：ALPHA_VECTOR_SCALING
// 启用时 alpha 必须为有效 device 向量（传 NULL 为参数错误）；BETA 同。
// 精确豁免域：ALPHA_VEC set 且 BETA_VEC 未 set 且 beta==nullptr —— 按
// NVIDIA 原文 "beta ... can have a NULL value only if ALPHA_VECTOR_SCALING
// is set without BETA_VECTOR_SCALING"，该组合合法（按 0.0，既有容忍路径）。
// 标量模式的 α/β nullptr 无条件容忍（已声明超集，维持不动）。
// vec 编码位执行期活读（与 fill_matmul_tiling_fields 同源，替代
// baseTd 的 PlanInit 快照）：PlanInit 后 SetAttribute 开 vec 时按活读状态
// 拦截 null/host 指针（快照路径下该组合静默落入标量分支、host 解引用
// device 指针崩溃——该崩溃域随活读消除）；后关 vec 时按标量分支放行。
// host 侧单次查询为 μs 级，相对 kernel 执行可忽略。
// alpha/beta 的对称校验见 validate_one_scaling_vector（顺序：先 16B
// 对齐后 device 判定）。
static aclsparseStatus_t validate_vector_scaling_pointers(const void* alpha, const void* beta,
                                                           const aclsparseLtMatmulDescriptor* md)
{
    const bool alphaVec = decode_alpha_scaling(md);
    const bool betaVec = decode_beta_scaling(md);
    // vec 模式 nullptr 拦截（α-vec×null / β-vec×null → INVALID）。
    if (alphaVec && alpha == nullptr) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: alpha vector scaling enabled but alpha is nullptr "
                "(requires a device float[M] vector)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (betaVec && beta == nullptr) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: beta vector scaling enabled but beta is nullptr "
                "(requires a device float[M] vector)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alphaVec && alpha != nullptr) {
        aclsparseStatus_t st = validate_one_scaling_vector(alpha, "alpha");
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    if (betaVec && beta != nullptr) {
        aclsparseStatus_t st = validate_one_scaling_vector(beta, "beta");
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// checked_mul_add — shared/aclsparselt_internal.h 共享 helper（本 TU 经
// include 引入，原本地重复定义已消除）。

// mat_batch_phys_span —— 单矩阵侧 V1 span 校验（自 validate_matmul_one_batch_side
// 拆出，codecheck 超大函数项，语句与返回语义逐位等价）：physSpan（元素单位，
// order 感知）。rows/cols/ld 为 int64 且描述符 setter 无上限校验，
// (rows/cols-1)*ld 乘加链先做受检查计算，溢出即布局非法（物理 span 超
// int64 无法被 kernel 寻址）——先于任何后续乘法。溢出时打带参日志并返回
// false（span 仅在返回 true 时有效）。
static bool mat_batch_phys_span(const aclsparseLtMatDescriptor* desc, const char* name,
                                int64_t& span)
{
    const bool spanOk = (desc->order == ACL_SPARSE_ORDER_COL)
        ? checked_mul_add(desc->cols - 1, desc->ld, desc->rows, span)
        : checked_mul_add(desc->rows - 1, desc->ld, desc->cols, span);
    if (!spanOk) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: mat%s physical span overflow (rows=%lld, cols=%lld, ld=%lld)",
                name, static_cast<long long>(desc->rows),
                static_cast<long long>(desc->cols), static_cast<long long>(desc->ld));
    }
    return spanOk;
}

// validate_matmul_one_batch_side — 单矩阵侧（A/B/C/D 之一）的 batch 布局
// V1/V2/V3 校验（V1 对 stride==0 的广播侧跳过，见下方判据处注释）。
static aclsparseStatus_t validate_matmul_one_batch_side(
    const aclsparseLtMatDescriptor* desc, const char* name,
    int32_t numBatches)
{
    const int32_t dt = dtype_from_acl(desc->valueType);
    if (dt == SPLT_DTYPE_INVALID) {
        return ACL_SPARSE_STATUS_SUCCESS;  // 非四主型：执行层 dtype gate 拒绝
    }
    const int64_t elemSize =
        (dt == SPLT_DTYPE_FP32 || dt == SPLT_DTYPE_INT32) ? SPLT_FP32_BYTES :
        (dt == SPLT_DTYPE_INT8) ? SPLT_INT8_BYTES : SPLT_HALF_BYTES;
    // V1: physSpan（元素单位，order 感知），见 mat_batch_phys_span。
    int64_t span = 0;
    if (!mat_batch_phys_span(desc, name, span)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const int64_t stride = desc->batchStride;
    // stride < 0 恒非法（kernel 按 base + b*stride 推进，负值回走越界）；
    // stride == 0 为跨批广播（对齐 cuSPARSELt batch_stride=0 语义：该侧所有
    // batch 读写 batch 0，b*0 恒为基址）——缓冲仅需单批容量，V1 span 校验
    // 跳过该侧；V2 对齐 0%x==0 恒过、V3 (nb-1)*0+span 无溢出面，二者对
    // 广播侧自然失效。各侧独立设置、独立判定。
    if (stride < 0) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: mat%s batchStride=%lld < 0 (numBatches=%d)",
                name, static_cast<long long>(stride), numBatches);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (stride == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;  // 广播：全批读写 batch 0
    }
    if (stride < span) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: mat%s batchStride=%lld < span=%lld (physSpan=%lld, numBatches=%d)",
                name, static_cast<long long>(stride), static_cast<long long>(span),
                static_cast<long long>(span), numBatches);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // V2: 逐 batch 基址 16B 对齐。无乘法等价判断：elemSize ∈ {1,2,4} 均为
    // 2 的幂，故 stride*elemSize ≡ 0 (mod 16) ⟺ stride ≡ 0 (mod 16/elemSize)
    // ——纯取模，不计算可溢出 int64 的 stride*elemSize 乘积；日志只打印
    // 操作数（打印乘积同样构成溢出 UB）。
    if (stride % (16 / elemSize) != 0) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: mat%s batchStride=%lld not 16-byte aligned (elemSize=%lld, requires multiple of %lld)",
                name, static_cast<long long>(stride), static_cast<long long>(elemSize),
                static_cast<long long>(16 / elemSize));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // V3: 溢出防护（乘加链，最终防线——V2 放行的对齐巨大 stride 由本层拦截）。
    int64_t totalElems = 0;
    int64_t totalBytes = 0;
    if (!checked_mul_add(numBatches - 1, stride, span, totalElems) ||
        !checked_mul_add(totalElems, elemSize, 0, totalBytes)) {
        OP_LOGE(kSparseLtLogTag, "Matmul: mat%s batch layout overflow (numBatches=%d, batchStride=%lld)",
                name, numBatches, static_cast<long long>(stride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_matmul_batch_strides —— numBatches > 1 时四矩阵的
// batchStride 防御校验（语义上拦截"kernel 按 stride 推进 batch 循环时
// 必然越界读"的非法输入，属"合法输入必产正确结果"的必要防御，非收紧
// 合法域）：
//   V1 batchStride > 0 且 ≥ 各自物理 span（order 感知：ROW 序
//      (rows-1)*ld+cols，COL 序 (cols-1)*ld+rows；span 乘加链经受检查
//      计算，溢出即布局非法）——(b) 下稀疏侧 A_pruned/B_pruned 为描述符
//      (order, ld) 布局输出，其 batch 访问跨度已被上述物理 span 精确覆盖；
//      batchStride == 0 为跨批广播（cuSPARSELt batch_stride=0 语义：该侧
//      全批读写 batch 0，缓冲仅需单批 span），该侧 V1 跳过；负值拒绝；
//   V2 batchStride 为 16/elemSize 的倍数（⟺ batchStride × elemSize 为
//      16 字节倍数——纯取模等价判断，不计算可溢出的乘积；配合基址 16B
//      校验，逐 batch 基址恒 16B 对齐）；
//   V3 (numBatches-1)*batchStride + span 与 ×elemSize 按 int64 受检查
//      乘加防溢出（最终防线：V2 放行的对齐巨大 stride 由本层拦截）。
// A/B 侧 span 按原始物理布局上界（kernel 经 GM 布局行距直读 ld，实际
// 访问为原始物理布局的逻辑子集，上界既不放宽拦截域也不误拒合法输入）。C/D 侧此处按 batch 交互
// 维度复检（stride ≥ span / 对齐 / 防溢出同 V1-V3）。
// 位置选择（prune V1 同模式）：执行期校验，覆盖 MatDescSetAttribute 在
// PlanInit 之后修改 batchStride 的任意时序（Matmul 与 Search 的候选执行
// 均经本函数）。
// numBatches 执行期活读（与 fill_matmul_tiling_fields 同源，替代
// baseTd 的 PlanInit 快照）：kernel batch 循环消费的是活读值，校验须与
// 之同源。
static aclsparseStatus_t validate_matmul_batch_strides(const aclsparseLtMatmulDescriptor* md)
{
    const int32_t numBatches = (md->matA != nullptr && md->matA->numBatches > 0)
        ? md->matA->numBatches : 1;
    if (numBatches <= 1) {
        return ACL_SPARSE_STATUS_SUCCESS;  // 单 batch 直通（行为与返回码不变）
    }
    struct BatchSide {
        const aclsparseLtMatDescriptor* desc;
        const char* name;
    };
    const BatchSide sides[4] = {
        {md->matA, "A"},
        {md->matB, "B"},
        {md->matC, "C"},
        {md->matD, "D"},
    };
    for (const BatchSide& side : sides) {
        if (side.desc == nullptr) { continue; }
        aclsparseStatus_t st = validate_matmul_one_batch_side(side.desc, side.name,
                                                              numBatches);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_bias_params —— bias 侧防御校验（X5/X6 同族范式的第 5 维
// 补齐，执行期，Matmul 与 Search 候选执行共用）：
//   B1 biasPointer 16 字节对齐（对齐 cuSPARSELt "All pointers must be
//      aligned to 16 bytes"）：bias 经 SpltLoadBiasFullToFp32 的
//      DataCopyPad 以 GM 源地址读取——实测 DataCopyPad 容忍 16B 对齐
//      GM 源（16B 非 32B 基址探针全过：bias 四 dtype fullload/chunk
//      路径，回归锚点见 ltmatmul_test_align16.cpp；原 32B 口径为历史
//      收紧，D5 轮实验证伪后回摆 16B）。splt_data_ptrs_aligned16
//      覆盖 workspace/matA/B/C/D 不含 bias；用户偏移后的非 16B 对齐
//      bias 指针由本校验转化为带日志的 host 侧报错（可诊断化）。
//   B2 biasStride >= 0：负值在 kernel batch 循环（b × biasStride）回走
//      越界；0 合法（全部 batch 共用同一 bias 的广播语义，头文件
//      BIAS_STRIDE 属性声明）。
//   B3 逐 batch bias 源地址 16 字节对齐：numBatches > 1 且 biasStride > 0
//      时，batch b 的 GM 源为 biasVecGM[b × biasStride]（DataCopyPad 同
//      B1 的 16B 口径，实测逐 batch 16B 源精度通过——biasStride=4 的
//      FP32 双 batch 探针为回归锚点）。无乘法等价判断（与 batch V2 同
//      范式）：biasElemSize ∈ {2,4} 均为 2 的幂，biasStride × biasElemSize
//      ≡ 0 (mod 16) ⟺ biasStride ≡ 0 (mod 16/biasElemSize)（bias 为
//      FP32 → %4==0；FP16/BF16 → %8==0）。bias dtype 键于 (A/B valueType,
//      C valueType)（SpltBiasDType<T, CType>：cuSPARSELt 规则 bias = C
//      dtype，INT8 × C/D ∈ {INT8,INT32}（32I 域）例外为 FP32——INT8 ×
//      C/D=FP16/BF16 的 bias 为 FP16/BF16，其 biasStride 合法域相应
//      收紧为 %8==0）。
// biasPointer == nullptr 时无消费方，直通。
static aclsparseStatus_t validate_bias_params(const aclsparseLtMatmulDescriptor* md)
{
    if (md->biasPointer == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if ((reinterpret_cast<uintptr_t>(md->biasPointer) % 16) != 0) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: biasPointer not 16-byte aligned (DataCopyPad GM source requirement)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (md->biasStride < 0) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: biasStride=%lld < 0 (0=broadcast, positive=per-batch stride)",
                static_cast<long long>(md->biasStride));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // B3：逐 batch 源地址 16B 对齐（见函数头注释）。
    const int32_t numBatches = (md->matA != nullptr && md->matA->numBatches > 0)
        ? md->matA->numBatches : 1;
    if (numBatches > 1 && md->biasStride > 0) {
        const int32_t dt = dtype_from_acl(md->matA->valueType);
        const int32_t cDt = dtype_from_acl(md->matC->valueType);
        // bias dtype 键于 (dt, cDt)（SpltBiasDType<T, CType> 口径）：
        // FP32 或 INT8 × 32I 域（C ∈ {INT8,INT32}）→ FP32；否则 = C dtype
        //（FP16/BF16，含 INT8 × C/D=FP16/BF16）。
        const bool biasIsFp32 = (dt == SPLT_DTYPE_FP32) ||
            (dt == SPLT_DTYPE_INT8 && (cDt == SPLT_DTYPE_INT8 || cDt == SPLT_DTYPE_INT32));
        const int64_t biasElemSize = biasIsFp32 ? SPLT_FP32_BYTES : SPLT_HALF_BYTES;
        if (md->biasStride % (16 / biasElemSize) != 0) {
            OP_LOGE(kSparseLtLogTag,
                    "Matmul: biasStride=%lld not 16-byte aligned per batch (biasElemSize=%lld, "
                    "requires multiple of %lld)",
                    static_cast<long long>(md->biasStride),
                    static_cast<long long>(biasElemSize),
                    static_cast<long long>(16 / biasElemSize));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_workspace_sufficiency —— 执行期 workspace 充分性校验
//（活读 NUM_BATCHES × PlanInit workspace 快照的失配防线）。
// 背景：temp 区（splitK>1）按 PlanInit 时点 numBatches 分配并计入
// plan->workspaceSize 快照；Y1 统一活读后 kernel 的 memset 与 cube 写按
// 活读 numBatches 推进（matmul_host 的 tempBytes / kernel 的
// b*splitK*m*n*4 偏移）——PlanInit 后增大 NUM_BATCHES（头文件 SetAttribute
// 时序契约明文允许）在 splitK>1 下将越界写穿 workspace 尾部。GetWorkspace
// 恒返回快照无法反映变更，故在执行入口按活读状态重算 WsLayout 对比
// 快照，超限返回 INSUFFICIENT_RESOURCES（与 Search 侧 plan-A trim
//（AppendPowerCandidates/AppendCurrentConfigFallback 的同判据）恢复防线
// 对称——Search 的候选 trim 保留，本函数补齐 Matmul 直调与 Search 候选
// 执行两路径）。
// 零影响锚点：正常时序（Init 后属性不变）下重算与 PlanInit 快照完全
// 同式同参（m/n/k/dtype 恒定、splitK 为 PlanInit 冻结值、活读 numBatches
// == 快照值），恒等通过；numBatches 减小 / BATCH_STRIDE 单独变更（temp
// 区不依赖 stride）亦恒通过；splitK==1（fused，默认计划）totalBytes 不含
// batch 项，天然免疫。Search 路径：候选执行前 RecomputePlan 已按同参
// 重写 plan->workspaceSize，重算值与之恒等（fused 候选 totalBytes ==
// tempOff ≤ 候选快照）。
static aclsparseStatus_t validate_workspace_sufficiency(
    const aclsparseLtMatmulDescriptor* md,
    const AclsparseltTilingData* baseTd,
    size_t planWorkspaceSize)
{
    // 活读 numBatches（与 fill_matmul_tiling_fields / kernel batch 循环同源）。
    const int32_t liveBatches = (md->matA != nullptr && md->matA->numBatches > 0)
        ? md->matA->numBatches : 1;
    const WsLayout wsl = compute_ws_layout(md_m(md), md_n(md), md_k(md),
                                           baseTd->splitK, baseTd->dataType, liveBatches,
                                           splt_apruned_span_elems(md, md_m(md), md_k(md)));
    if (static_cast<int64_t>(wsl.totalBytes) > static_cast<int64_t>(planWorkspaceSize)) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: workspace insufficient under live state: need=%lld bytes "
                "(numBatches=%d, splitK=%d) > plan workspaceSize=%lld. NUM_BATCHES was "
                "increased after PlanInit; re-run PlanInit (and reallocate workspace) "
                "or restore numBatches",
                static_cast<long long>(wsl.totalBytes), liveBatches, baseTd->splitK,
                static_cast<long long>(planWorkspaceSize));
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_scaling_ub_budget —— two-stage（splitK>1）路径的向量族
// UB 预算校验（Y2 同范式可诊断拒绝）。
// 背景：alpha/beta 缩放向量与 bias 均经 SpltInitBiasBuffers /
// SpltLoadScalingUB 按全量 m*4（bias FP16/BF16 另有 m*2 原始缓冲）驻留
// UB，two-stage epilogue 的其余缓冲为 CHUNK=256 固定小缓冲——向量族字节
// 与 m 线性相关且无分块，m 超过 UB 预算阈值时 InitBuffer 超 UB（启动
// 失败/断言，取决于 CANN 版本）。fused 路径（splitK==1）经
// compute_bias_chunk_mode 活读重算降级为 per-tile 惰性加载（三向量同
// 策略，降级后 UB 需求为 tile 级常量恒可容纳）；two-stage 的 epilogue
// 逐 piece 消费向量（单行单标量），惰性加载退化为逐 piece 4 字节 GM 事
// 务（性能不可接受），且 splitK>1 域本身要求 m*n*splitK*4 字节 workspace
// （阈值 m 下已达数十 MB），该域可达面极窄——故按 Y2 范式在 host 侧
// 转化为带日志的可诊断拒绝（NOT_SUPPORTED）。
static aclsparseStatus_t validate_scaling_ub_budget(
    const aclsparseLtMatmulDescriptor* md,
    const AclsparseltTilingData* baseTd)
{
    if (baseTd->splitK <= 1) {
        return ACL_SPARSE_STATUS_SUCCESS;  // fused 路径经 biasChunkMode 降级覆盖
    }
    const bool alphaVec = decode_alpha_scaling(md);
    const bool betaVec = decode_beta_scaling(md);
    const bool biasEnabled = (md->biasPointer != nullptr);
    if (!alphaVec && !betaVec && !biasEnabled) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    const int32_t m = md_m(md);
    const int32_t dt = baseTd->dataType;
    const int32_t cDt = baseTd->cDataType;
    // biasLoadBuf：FP16/BF16 bias 的原始加载缓冲（bias 为 FP32 直载时无
    // 中间缓冲）——与 compute_bias_chunk_mode 的口径一致（键于 (dt, cDt)：
    // INT8 × C/D=FP16/BF16 的 bias 为 FP16/BF16，需计 m*2）。
    const bool biasIsFp32 = (dt == SPLT_DTYPE_FP32) ||
        (dt == SPLT_DTYPE_INT8 && (cDt == SPLT_DTYPE_INT8 || cDt == SPLT_DTYPE_INT32));
    const bool needBiasLoadBuf = biasEnabled && !biasIsFp32;
    const uint64_t vecBytes =
        (alphaVec ? static_cast<uint64_t>(m) * 4 : 0) +
        (betaVec ? static_cast<uint64_t>(m) * 4 : 0) +
        (biasEnabled ? static_cast<uint64_t>(m) * 4 : 0) +
        (needBiasLoadBuf ? static_cast<uint64_t>(m) * static_cast<uint64_t>(SPLT_HALF_BYTES)
                         : 0);
    // two-stage epilogue 固定 chunk 缓冲组上界（SpltEpilogueInitBuffers：
    // acc/temp/dFp32 各 1KB + dTBuf/halfBuf/cTBuf/cFp32Buf <=1KB 级 +
    // COL wideCdBuf 8KB + reluMask 64B），保守取 16KB。
    constexpr uint64_t kTwoStageChunkBytes = 16 * 1024;
    const uint64_t ubSize = get_ub_size();
    if (vecBytes + kTwoStageChunkBytes > ubSize) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: vector scaling/bias UB budget exceeded on two-stage path: "
                "vecBytes=%llu + chunkBufs=%llu > ubSize=%llu (m=%d, splitK=%d, "
                "alphaVec=%d, betaVec=%d, bias=%d). Per-vector full-load requires "
                "m*4 bytes each; use splitK=1 (fused, per-tile chunked load) or "
                "reduce m",
                static_cast<unsigned long long>(vecBytes),
                static_cast<unsigned long long>(kTwoStageChunkBytes),
                static_cast<unsigned long long>(ubSize), m, baseTd->splitK,
                alphaVec ? 1 : 0, betaVec ? 1 : 0, biasEnabled ? 1 : 0);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t prepare_matmul_pointers(
    const aclsparseLtMatmulDescriptor* md,
    const AclsparseltTilingData* baseTd,
    const void* alpha, const void* beta,
    const void* matA, const void* matB, const void* matC, void* matD,
    void* workspace, aclrtStream stream,
    MatmulLaunchArgs& args)
{
    aclsparseStatus_t vsRet = validate_vector_scaling_pointers(alpha, beta, md);
    if (vsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return vsRet;
    }
    aclsparseStatus_t bsRet = validate_matmul_batch_strides(md);
    if (bsRet != ACL_SPARSE_STATUS_SUCCESS) {
        return bsRet;
    }
    aclsparseStatus_t biasRet = validate_bias_params(md);
    if (biasRet != ACL_SPARSE_STATUS_SUCCESS) {
        return biasRet;
    }
    // two-stage（splitK>1）向量族 UB 预算校验（fused 路径经
    // compute_bias_chunk_mode 降级覆盖，不在此拦截）。
    aclsparseStatus_t ubRet = validate_scaling_ub_budget(md, baseTd);
    if (ubRet != ACL_SPARSE_STATUS_SUCCESS) {
        return ubRet;
    }
    fill_matmul_tiling_fields(md, baseTd, alpha, beta, args);

    // GM 布局接入（(b) 两侧统一）：COL 转置视图的 cube 转置标志翻转
    //（ld 直通 td.ldA/td.ldB，无需预处理）须先于 H2D push
    //（下方 push_tiling_to_device）。
    aclsparseStatus_t st = splt_apply_gm_layout(md, args.td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = push_tiling_to_device(workspace, args.td.tilingOffset, args.td);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    st = resolve_matmul_pointers(md, matA, matB, matC, matD, workspace, args);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// validate_launch_dtypes — launch 前 dtype 配对白名单（Defense-in-depth）：
//   1) 主 dtype 合法域（FP32/FP16/BF16/INT8）——非法值不得触达 kernel
//      分派 switch；
//   2) C/D dtype 配对白名单——kernel 分派以 cDataType 为键且 OutType 与
//      CType 绑定（fused/epilogue 的 int8_ci32/int8_cf16/int8_cbf16 按
//      C dtype 的元素宽度读写 D），域外组合会产生越界写（如 C=FP16、
//      D=INT8 按 2B 写 1B 声明的 matD）或静默错值，统一转化为带日志的
//      NOT_SUPPORTED：
//      - INT8 A/B：(C,D) ∈ {(INT8,INT8),(INT8,INT32),(INT32,INT32),
//        (FP16,FP16),(BF16,BF16)}（NVIDIA dtype 表五组合）；
//      - 非 INT8 A/B（FP32/FP16/BF16）：C == D == dataType（同 dtype）。
//      描述符层（matmulDescriptor）保持库基线零 dtype 校验（超集组合
//      在 DescriptorInit 被接受），本校验在执行层拦截（Matmul/Search
//      同一 spltMatmulRunInternal 路径）。
static aclsparseStatus_t validate_launch_dtypes(const MatmulLaunchArgs& args)
{
    if (args.dt != SPLT_DTYPE_FP32 && args.dt != SPLT_DTYPE_FP16 &&
        args.dt != SPLT_DTYPE_BF16 && args.dt != SPLT_DTYPE_INT8) {
        OP_LOGE(kSparseLtLogTag, "Matmul: invalid dataType=%d (only FP32/FP16/BF16/INT8)", (int)args.dt);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t cDt = args.td.cDataType;
    const int32_t dDt = args.td.outDataType;
    bool pairOk = false;
    if (args.dt == SPLT_DTYPE_INT8) {
        // NVIDIA dtype 表五组合：C=D=INT8 / C=INT8,D=INT32 / C=D=INT32 /
        // C=D=FP16 / C=D=BF16（kernel 分派 int8 / int8_i32 / int8_ci32 /
        // int8_cf16 / int8_cbf16 一一对应）。
        pairOk = (cDt == SPLT_DTYPE_INT8 && dDt == SPLT_DTYPE_INT8) ||
                 (cDt == SPLT_DTYPE_INT8 && dDt == SPLT_DTYPE_INT32) ||
                 (cDt == SPLT_DTYPE_INT32 && dDt == SPLT_DTYPE_INT32) ||
                 (cDt == SPLT_DTYPE_FP16 && dDt == SPLT_DTYPE_FP16) ||
                 (cDt == SPLT_DTYPE_BF16 && dDt == SPLT_DTYPE_BF16);
    } else {
        // 非 INT8：C/D 须与 A/B 同 dtype（epilogue 按 T=CType=OutType 读写）。
        pairOk = (cDt == args.dt && dDt == args.dt);
    }
    if (!pairOk) {
        OP_LOGE(kSparseLtLogTag,
                "Matmul: unsupported C/D dtype pair: dataType=%d requires %s, got "
                "(cDataType=%d, outDataType=%d)",
                (int)args.dt,
                (args.dt == SPLT_DTYPE_INT8)
                    ? "(C,D) in {(INT8,INT8),(INT8,INT32),(INT32,INT32),(FP16,FP16),(BF16,BF16)}"
                    : "C == D == dataType",
                (int)cDt, (int)dDt);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// launch_matmul_kernels — fused (splitK==1) vs splitK>1 (matmul ->
// epilogue) kernel launch.（B4：temp 的全量 aclrtMemsetAsync 已完全去除——
// 正确性不再依赖 memset，见下方不变量注释；launch 序列仅 cube → epilogue。）
// epilogue/fused launchers take outDataType (drives INT8 INT8-vs-INT32 dispatch).
// dtype 域守卫见 validate_launch_dtypes（拆出自本函数，阈值/条件逐位不变）。
static aclsparseStatus_t launch_matmul_kernels(const MatmulLaunchArgs& args, aclrtStream stream)
{
    aclsparseStatus_t dtRet = validate_launch_dtypes(args);
    if (dtRet != ACL_SPARSE_STATUS_SUCCESS) {
        return dtRet;
    }
    if (args.td.splitK == 1) {
        // Fused path: single __mix__(1,2) kernel.
        // AIC: matmul -> Fixpipe L0C->UB -> CrossCore notify AIV
        // AIV: Vector(alpha*acc+beta*C+Cast) -> DataCopyPad UB->GM(D)
        splt_fused_matmul_kernel_launch(reinterpret_cast<GM_ADDR>(args.aPruned),
                                        reinterpret_cast<GM_ADDR>(args.bMat),
                                        reinterpret_cast<GM_ADDR>(args.cMat),
                                        reinterpret_cast<GM_ADDR>(args.dMat),
                                        reinterpret_cast<GM_ADDR>(args.tilingGm),
                                        args.dt, args.td.cDataType, args.td.outDataType,
                                        static_cast<uint32_t>(args.td.usedCoreNum), stream);
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    // Slow path: two-segment (cube->temp->epilogue reduce) for splitK>1.
    {
        // [B4] temp memset fully removed (previously: a full
        // aclrtMemsetAsync of splitK*m*n*batches*4 bytes on every execution,
        // with correctness strongly coupled to the memset — historically
        // caused a precision bug from "clearing batch 0 only"). Correctness
        // invariant (five-point proof, plan section 2.3):
        //   1. The epilogue reads only segments [0, validSplitK)
        //      (SpltEpilogueBatchLoop's reduction upper bound now sources
        //      td.validSplitK, and the empty segments in [validSplitK, splitK)
        //      are never read because each such segment start, the product
        //      s×kSegLen, has already reached k).
        //   2. For every valid segment s (its start index kSegStart lies
        //      below k): the cube side executes every tile of that segment
        //      in the mTile and nTile dimensions (the continue meant for
        //      segments whose kSegStart has reached k never fires), and the
        //      tile grid mTiles×nTiles (mTiles being the ceiling of m over
        //      baseM, nTiles the ceiling of n over baseN) covers all of (m,n).
        //   3. Every tile performs at least one Mmad: SpltKL1Loop's first
        //      iteration has kPosL1 at kSegStart, still below k, hence
        //      curKL1 is at least one and L0C is initialized with the
        //      cmatrixInitVal flag set to true (no stale values remain).
        //   4. Fixpipe writes curM*curN fully to GM temp (including the m/n
        //      tail-tile shrinkage), so the written region covers all m*n
        //      elements of that segment.
        //   5. Symmetry across the batch dim: the cube writes all valid
        //      segments to every per-b offset tempBatchGm, and the epilogue
        //      reads every b — the coverage is symmetric.
        // Counterexample domain (if ever introduced, memset or an equivalent
        // must be restored): the cube skipping the write of any (m,n) tile
        // (degenerate shapes with a zero m or n dimension are already
        // excluded by the totalTiles non-positive early return and by
        // positive descriptor rows/cols), or the epilogue's read domain
        // extending beyond validSplitK.
        // This invariant is anchored alongside at the epilogue reduction
        // re-source site (SpltEpilogueBatchLoop).
        //
        // Keep the 8GiB sanity check on tempBytes (an independent validation
        // block anchoring the workspace temp-region usage — same caliber as
        // the allocation side compute_ws_layout, whose tempBytes is the
        // splitK*m*n*batches footprint measured in SPLT_FP32_BYTES units).
        const size_t numBatches = static_cast<size_t>(args.td.numBatches > 0 ? args.td.numBatches : 1);
        const size_t tempBytes = static_cast<size_t>(args.td.splitK) *
                                 static_cast<size_t>(args.td.m) *
                                 static_cast<size_t>(args.td.n) *
                                 numBatches * SPLT_FP32_BYTES;
        // Guard against unreasonable allocation (cap at 8 GiB temp buffer).
        constexpr size_t SPLT_TEMP_BYTES_CAP = static_cast<size_t>(8) * 1024 * 1024 * 1024;
        if (tempBytes > SPLT_TEMP_BYTES_CAP) {
            OP_LOGE(kSparseLtLogTag, "Matmul: temp buffer overflow, tempBytes=%zu > cap=%zu",
                    tempBytes, SPLT_TEMP_BYTES_CAP);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    splt_matmul_kernel_launch(reinterpret_cast<GM_ADDR>(args.aPruned),
                              reinterpret_cast<GM_ADDR>(args.bMat),
                              reinterpret_cast<GM_ADDR>(args.temp),
                              reinterpret_cast<GM_ADDR>(args.tilingGm), args.dt,
                              static_cast<uint32_t>(args.td.usedCoreNum), stream);
    splt_epilogue_kernel_launch(reinterpret_cast<GM_ADDR>(args.temp),
                                reinterpret_cast<GM_ADDR>(args.cMat),
                                reinterpret_cast<GM_ADDR>(args.dMat),
                                reinterpret_cast<GM_ADDR>(args.tilingGm), args.dt,
                                args.td.cDataType, args.td.outDataType,
                                static_cast<uint32_t>(args.td.usedCoreNum), stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// spltMatmulRunInternal —— 公共 aclsparseLtMatmul 的执行核心。
// 校验链与 kernel launch 顺序与公共入口完全一致（同一 prepare 路径）。
// 声明于 matmul_kernel.h（matmul_search_host.cpp 经 TimeCandidate 复用本入口）。
aclsparseStatus_t spltMatmulRunInternal(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams)
{
    aclsparseLtMatmulDescriptor* md = nullptr;
    AclsparseltTilingData* baseTd = nullptr;
    aclrtStream stream = nullptr;
    aclsparseStatus_t st = validate_matmul_params(handle, plan, streams, numStreams, workspace,
                                                   matA, matB, matC, matD, md, baseTd, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // 执行期 workspace 充分性校验：按活读 numBatches 重算布局对比
    // PlanInit 快照，超限拒绝（防止 splitK>1 × 活读增大 batch 的 temp 区
    // 越界写；正常时序重算恒等于快照，零影响）。
    st = validate_workspace_sufficiency(md, baseTd, (*plan)->workspaceSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    MatmulLaunchArgs args;
    st = prepare_matmul_pointers(md, baseTd, alpha, beta, matA, matB, matC, matD, workspace,
                                 stream, args);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    return launch_matmul_kernels(args, stream);
}

extern "C" aclsparseStatus_t aclsparseLtMatmul(
    const aclsparseLtHandle_t* handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams)
{
    return spltMatmulRunInternal(handle, plan, alpha, matA, matB, beta, matC, matD,
                                 workspace, streams, numStreams);
}
