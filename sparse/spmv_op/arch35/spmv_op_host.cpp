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
 * \file spmv_op_host.cpp
 * \brief spmv_op Host 侧实现：7 个 Generic API 入口。
 *
 * API 清单：
 *   F1. aclsparseSpMVOp_bufferSize
 *   F2. aclsparseSpMVOp_createDescr
 *   F3. aclsparseSpMVOp_destroyDescr
 *   F4. aclsparseSpMVOp_createPlan
 *   F5. aclsparseSpMVOp_destroyPlan
 *   F6. aclsparseSpMVOp_setGlobalUserData
 *   F7. aclsparseSpMVOp（主执行入口）
 *
 * 每个 API 内部拆分为 ValidateParams + LaunchKernel 两个 static 函数。
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "spmv_op.h"
#include "spmv_op_tiling_data.h"
// Host 侧不引入 kernel_operator.h，此处提供 GM_ADDR 的 host 编译回退定义。
// NPU 侧由 toolkit (kernel_utils_macros.h) 自动定义为 __gm__ uint8_t*。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif
#include "spmv_op_kernel.h"

// Tag 常量，用于 dlog 的模块标识
static constexpr const char *kTag = "aclsparseSpMVOp";

namespace {

// Internal plan struct (opaque to external code; public type is void*)
// 生命周期约束：descr 必须先于 plan 销毁，即用户调用顺序必须为：
//   createDescr → createPlan → execute... → destroyPlan → destroyDescr
// 若违反此顺序（先 destroyDescr），plan->descr 将成为悬垂指针，行为未定义。
struct SpmvOpPlanData {
    aclsparseSpMVOpDescr *descr{nullptr};  // weak ref，descr 必须先于 plan 销毁
    aclsparseSpMVOpAlg_t alg{ACL_SPARSE_SPMVOP_ALG_DEFAULT};
};

// ===========================================================================
// 公共参数校验辅助函数
// ===========================================================================

// 校验 matA 描述符：非空、CSR 格式、FP32 值类型、rowOffsetType∈{I32,I64}、colIndType==I32
static aclsparseStatus_t SpmvOpValidateSpMat(
    const char *api, aclsparseConstSpMatDescr_t matA)
{
    if (matA == nullptr) {
        OP_LOGE(api, "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *inner = SpmvOpToMatInner(matA);
    if (inner->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE(api, "unsupported format %d (CSR required)",
                static_cast<int>(inner->format));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->valueType != ACL_FLOAT) {
        OP_LOGE(api, "unsupported valueType %d (FP32 required)",
                static_cast<int>(inner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->IdxType != ACL_SPARSE_INDEX_32I) {
        // SpMVOp: rowOffsetType 可为 I32/I64，colIndType (IdxType) 必须为 I32
        OP_LOGE(api, "unsupported colIndType (IdxType) %d (I32 required)",
                static_cast<int>(inner->IdxType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->ptrType != ACL_SPARSE_INDEX_32I &&
        inner->ptrType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(api, "unsupported rowOffsetType %d (I32/I64 only)",
                static_cast<int>(inner->ptrType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 opA 和 computeType（适用于 bufferSize/createDescr）
static aclsparseStatus_t SpmvOpValidateCommon(
    const char *api, aclsparseOperation_t opA, aclDataType computeType)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        OP_LOGE(api, "unsupported opA %d (NON_TRANSPOSE only)",
                static_cast<int>(opA));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (computeType != ACL_FLOAT) {
        OP_LOGE(api, "unsupported computeType %d (ACL_FLOAT only)",
                static_cast<int>(computeType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 alg 取值
static aclsparseStatus_t SpmvOpValidateAlg(const char *api, aclsparseSpMVOpAlg_t alg)
{
    if (alg != ACL_SPARSE_SPMVOP_ALG_DEFAULT &&
        alg != ACL_SPARSE_SPMVOP_ALG1 &&
        alg != ACL_SPARSE_SPMVOP_ALG2) {
        OP_LOGE(api, "invalid alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}




// ===========================================================================
// bufferSize 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMVOpBufferParams(
    aclsparseOperation_t opA, aclsparseConstSpMatDescr_t matA,
    aclDataType computeType, aclsparseSpMVOpAlg_t alg, size_t *bufferSize)
{
    aclsparseStatus_t st = SpmvOpValidateCommon(kTag, opA, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmvOpValidateSpMat(kTag, matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmvOpValidateAlg(kTag, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (bufferSize == nullptr) {
        OP_LOGE(kTag, "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// createDescr 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMVOpCreateDescrParams(
    aclsparseSpMVOpDescr_t *descr, aclsparseConstSpMatDescr_t matA,
    aclsparseSpMVOpAlg_t alg, void *buffer)
{
    if (descr == nullptr) {
        OP_LOGE(kTag, "descr output is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = SpmvOpValidateSpMat(kTag, matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmvOpValidateAlg(kTag, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    // ALG2: buffer 必须非空（createDescr 预处理写入 reorder/bin_edge）
    // ALG1: buffer 可为 NULL
    if (alg == ACL_SPARSE_SPMVOP_ALG2) {
        if (buffer == nullptr) {
            OP_LOGE(kTag, "buffer is nullptr (ALG2 requires buffer)");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// createPlan 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMVOpCreatePlanParams(
    aclsparseSpMVOpDescr_t descr, aclsparseSpMVOpPlan_t *plan,
    const void *epilogueLTOBuffer, size_t epilogueLTOBufferSize)
{
    if (descr == nullptr) {
        OP_LOGE(kTag, "descr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (plan == nullptr) {
        OP_LOGE(kTag, "plan output is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // NPU 侧不支持 LTO-IR：若传入非 NULL 则返回 NOT_SUPPORTED
    if (epilogueLTOBuffer != nullptr || epilogueLTOBufferSize > 0) {
        OP_LOGE(kTag, "epilogue LTO not supported on NPU (buffer=%p, size=%zu)",
                epilogueLTOBuffer, epilogueLTOBufferSize);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// execute 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateExecuteBasicParams(
    const SpmvOpPlanData *plan, const void *alpha, const void *beta,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ)
{
    if (plan == nullptr) {
        OP_LOGE(kTag, "plan is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (plan->descr == nullptr) {
        OP_LOGE(kTag, "plan->descr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE(kTag, "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(kTag, "beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecX == nullptr || vecY == nullptr || vecZ == nullptr) {
        OP_LOGE(kTag, "vecX/Y/Z is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateExecuteVectors(
    aclsparseHandle_t handle,
    const aclsparseSpMVOpDescr *matInner,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ, const void *beta)
{
    auto *vecXInner = SpmvOpToVecInner(vecX);
    auto *vecYInner = SpmvOpToVecInner(vecY);
    auto *vecZInner = SpmvOpToVecInner(vecZ);
    if (vecXInner->valueType != ACL_FLOAT ||
        vecYInner->valueType != ACL_FLOAT ||
        vecZInner->valueType != ACL_FLOAT) {
        OP_LOGE(kTag, "vecX/Y/Z valueType must be FP32");
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int64_t mInt = static_cast<int64_t>(matInner->m);
    const int64_t nInt = static_cast<int64_t>(matInner->n);
    if (nInt > 0 && vecXInner->values == nullptr) {
        OP_LOGE(kTag, "vecX.values is nullptr (n=%ld > 0)", nInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *h = SpmvOpToInternalHandle(handle);
    bool skipYCheck = false;
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST && beta != nullptr) {
        skipYCheck = (*reinterpret_cast<const float *>(beta) == 0.0f);
    } else if (h->pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        // DEVICE mode 无法在 host 侧读取 beta 值，kernel 内部有 beta==0 短路保护，
        // 故跳过 Y 的 values 非空校验（信任用户，文档约定 beta==0 时 Y 可为 null）
        skipYCheck = true;
    }
    if (!skipYCheck && mInt > 0 && vecYInner->values == nullptr) {
        OP_LOGE(kTag, "vecY.values is nullptr (m=%ld > 0, beta != 0)", mInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (mInt > 0 && vecZInner->values == nullptr) {
        OP_LOGE(kTag, "vecZ.values is nullptr (m=%ld > 0)", mInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecXInner->nums != static_cast<uint64_t>(nInt)) {
        OP_LOGE(kTag, "vecX.size=%zu != n=%ld", static_cast<size_t>(vecXInner->nums), nInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecYInner->nums != static_cast<uint64_t>(mInt)) {
        OP_LOGE(kTag, "vecY.size=%zu != m=%ld", static_cast<size_t>(vecYInner->nums), mInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (vecZInner->nums != static_cast<uint64_t>(mInt)) {
        OP_LOGE(kTag, "vecZ.size=%zu != m=%ld", static_cast<size_t>(vecZInner->nums), mInt);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpMVOpExecuteParams(
    aclsparseHandle_t handle,
    const SpmvOpPlanData *plan, const void *alpha, const void *beta,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ)
{
    aclsparseStatus_t st = ValidateExecuteBasicParams(plan, alpha, beta, vecX, vecY, vecZ);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    return ValidateExecuteVectors(handle, plan->descr, vecX, vecY, vecZ, beta);
}

// ===========================================================================
// execute kernel launch
// ===========================================================================

static aclsparseStatus_t ComputeSpmvOpBlockSplits(
    int32_t m, uint32_t &useBlocks, uint32_t &rowsPerBlock)
{
    if (m <= 0) {
        useBlocks = 0;
        rowsPerBlock = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(kTag, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    useBlocks = std::min(aivCoreNum,
        CeilDiv<uint32_t>(static_cast<uint32_t>(m), kSpmvOpMaxThreadsPerBlock));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    rowsPerBlock = CeilDiv<uint32_t>(static_cast<uint32_t>(m), useBlocks);
    if (rowsPerBlock == 0) {
        rowsPerBlock = 1;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static SpmvOpTilingData BuildSpmvOpTiling(
    const aclsparseSpMVOpDescr *descr,
    const aclsparseContext *h,
    const void *alpha, const void *beta,
    uint32_t rowsPerBlock)
{
    SpmvOpTilingData tiling{};
    tiling.m = static_cast<int32_t>(descr->m);
    tiling.indexBase = static_cast<int32_t>(descr->indexBase);
    tiling.rowOffsetType = (descr->rowOffsetType == ACL_SPARSE_INDEX_32I)
        ? SPMV_OP_IDX_RT_I32 : SPMV_OP_IDX_RT_I64;
    tiling.algType = (descr->alg == ACL_SPARSE_SPMVOP_ALG2)
        ? SPMV_OP_ALG_TYPE_2 : SPMV_OP_ALG_TYPE_1;
    // HOST mode: alpha/beta 是 host 指针，直接解引用，值通过 tiling 传给 kernel
    // DEVICE mode: alpha/beta 是 device 指针，kernel 直接从 device 内存读取，无需 D2H 中转
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        tiling.alpha = (alpha != nullptr) ? *reinterpret_cast<const float *>(alpha) : 0.0f;
        tiling.beta  = (beta  != nullptr) ? *reinterpret_cast<const float *>(beta)  : 0.0f;
        tiling.alphaPtr = 0ULL;
        tiling.betaPtr  = 0ULL;
    } else {
        tiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling.betaPtr  = reinterpret_cast<uint64_t>(beta);
    }
    tiling.rowsPerBlock = rowsPerBlock;
    // reorder / bin_edge 地址通过 kernel __gm__ 参数直接传入，不放入 tiling
    return tiling;
}

// ===========================================================================
// nnz=0 快捷路径：Z = beta * Y（NPU async kernel）
// ===========================================================================

static aclsparseStatus_t LaunchSpMVOpBetaYKernel(
    aclsparseHandle_t handle, const SpmvOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnVecDescr_t vecY, aclsparseDnVecDescr_t vecZ)
{
    auto *h = SpmvOpToInternalHandle(handle);
    aclrtStream stream = h->stream;
    const auto *descr = plan->descr;
    if (descr->m > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "BetaY: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(descr->m));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(descr->m);

    uint32_t useBlocks = 0;
    uint32_t rowsPerBlock = 0;
    aclsparseStatus_t splitSt = ComputeSpmvOpBlockSplits(m, useBlocks, rowsPerBlock);
    CHECK_RET(splitSt == ACL_SPARSE_STATUS_SUCCESS, return splitSt);

    // 构造 tiling（仅 beta/m/rowsPerBlock 有效）
    SpmvOpTilingData tiling{};
    tiling.m = m;
    tiling.rowsPerBlock = rowsPerBlock;
    // HOST mode: beta 是 host 指针，直接解引用
    // DEVICE mode: beta 是 device 指针，kernel 从 GM 读，无需同步
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        tiling.beta = (beta != nullptr) ? *reinterpret_cast<const float *>(beta) : 0.0f;
        tiling.alpha = (alpha != nullptr) ? *reinterpret_cast<const float *>(alpha) : 0.0f;
        tiling.alphaPtr = 0ULL;
        tiling.betaPtr = 0ULL;
    } else {
        tiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling.betaPtr = reinterpret_cast<uint64_t>(beta);
    }

    auto *yInner = SpmvOpToVecInner(vecY);
    auto *zInner = SpmvOpToVecInner(vecZ);

    spmv_op_beta_y_kernel_do(
        reinterpret_cast<GM_ADDR>(yInner->values),
        reinterpret_cast<GM_ADDR>(zInner->values),
        tiling, useBlocks, stream);

    OP_LOGI(kTag, "beta_y kernel launched: m=%d, numBlocks=%u", m, useBlocks);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 主 kernel launch: Z = alpha * A * X + beta * Y（nnz > 0 时调用）
// ===========================================================================

static aclsparseStatus_t LaunchSpMVOpNnzKernel(
    aclsparseHandle_t handle, const SpmvOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ,
    int32_t m, uint64_t nnz)
{
    auto *h = SpmvOpToInternalHandle(handle);
    const auto *descr = plan->descr;
    uint32_t useBlocks = 0, rowsPerBlock = 0;
    const bool isAlg2 = (descr->alg == ACL_SPARSE_SPMVOP_ALG2);
    if (isAlg2) {
        if (descr->alg2NumBlocks == 0) {
            OP_LOGE(kTag, "ALG2: alg2NumBlocks not set in descr");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        useBlocks = descr->alg2NumBlocks;
        rowsPerBlock = (m + useBlocks - 1) / useBlocks;
        if (rowsPerBlock == 0u) { rowsPerBlock = 1u; }
    } else {
        aclsparseStatus_t splitSt = ComputeSpmvOpBlockSplits(m, useBlocks, rowsPerBlock);
        CHECK_RET(splitSt == ACL_SPARSE_STATUS_SUCCESS, return splitSt);
    }
    SpmvOpTilingData tiling = BuildSpmvOpTiling(descr, h, alpha, beta, rowsPerBlock);
    auto *xInner = SpmvOpToVecInner(vecX);
    auto *yInner = SpmvOpToVecInner(vecY);
    auto *zInner = SpmvOpToVecInner(vecZ);
    auto *gmRowOffsets = reinterpret_cast<GM_ADDR>(descr->csrRowOffsets);
    auto *gmColInd = reinterpret_cast<GM_ADDR>(descr->csrColInd);
    auto *gmValues = reinterpret_cast<GM_ADDR>(descr->csrValues);
    GM_ADDR gmReorder = nullptr;
    GM_ADDR gmBinEdge = nullptr;
    if (tiling.algType == SPMV_OP_ALG_TYPE_2 && descr->userBuffer != nullptr) {
        auto *wsBase = static_cast<uint8_t *>(descr->userBuffer);
        SpmvOpWorkspaceLayout layout{};
        layout.Compute(m, useBlocks);
        gmReorder = reinterpret_cast<GM_ADDR>(wsBase + layout.reorderOffset);
        gmBinEdge = reinterpret_cast<GM_ADDR>(wsBase + layout.binEdgeOffset);
    }
    spmv_op_kernel_do(
        gmRowOffsets, gmColInd, gmValues,
        reinterpret_cast<GM_ADDR>(xInner->values),
        reinterpret_cast<GM_ADDR>(yInner->values),
        reinterpret_cast<GM_ADDR>(zInner->values),
        gmReorder, gmBinEdge, tiling, useBlocks, h->stream);
    OP_LOGI(kTag, "spmv_op kernel launched: m=%d, nnz=%llu, numBlocks=%u, alg=%d, rowOffType=%s",
            m, static_cast<unsigned long long>(nnz), useBlocks,
            static_cast<int>(tiling.algType),
            (tiling.rowOffsetType == SPMV_OP_IDX_RT_I32) ? "i32" : "i64");
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchSpMVOpKernel(
    aclsparseHandle_t handle, const SpmvOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ)
{
    const auto *descr = plan->descr;
    if (descr->m > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(descr->m));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(descr->m);
    if (m == 0) {
        OP_LOGD(kTag, "m=0, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->nnz == 0) {
        return LaunchSpMVOpBetaYKernel(handle, plan, alpha, beta, vecY, vecZ);
    }
    return LaunchSpMVOpNnzKernel(handle, plan, alpha, beta, vecX, vecY, vecZ, m, descr->nnz);
}

}  // namespace

// ===========================================================================
// Public APIs
// ===========================================================================
extern "C" {

// F1: bufferSize
aclsparseStatus_t aclsparseSpMVOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ,
    aclDataType computeType,
    aclsparseSpMVOpAlg_t alg,
    size_t *bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "bufferSize: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMVOpBufferParams(
        opA, matA, computeType, alg, bufferSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    auto *matInner = SpmvOpToMatInner(matA);

    // m > INT32_MAX 时截断，提前返回 NOT_SUPPORTED（避免 silent truncation）
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "bufferSize: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(matInner->rows));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(matInner->rows);

    if (alg == ACL_SPARSE_SPMVOP_ALG2) {
        // ALG2 需要 workspace：先估算 numBlocks 以计算 bin_edge 大小
        uint32_t useBlocks = 0;
        uint32_t rowsPerBlock = 0;
        aclsparseStatus_t splitSt = ComputeSpmvOpBlockSplits(m, useBlocks, rowsPerBlock);
        if (splitSt != ACL_SPARSE_STATUS_SUCCESS) { return splitSt; }
        SpmvOpWorkspaceLayout layout{};
        layout.Compute(m, useBlocks);
        *bufferSize = layout.totalBytes;
    } else {
        // ALG1 / DEFAULT: 无 workspace
        *bufferSize = 0;
    }

    OP_LOGD(kTag, "bufferSize: m=%d, alg=%d, bufferSize=%zu", m, static_cast<int>(alg), *bufferSize);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ALG2 预处理：异步 launch device-side preprocess kernel
// 预处理 kernel 在 device 上完成 rowOffsets 读取、按 nnz 降序排序以及
// bin_edge 负载均衡切分。kernel 与后续 SpMVOp 共用同一 stream，stream
// 的 FIFO 语义保证预处理先于计算完成，无需 aclrtSynchronizeStream。
// newDescr 为借用指针，生命周期和失败路径清理由调用方统一管理。
static aclsparseStatus_t CreateDescrPreprocessAlg2(
    aclsparseHandle_t handle, aclsparseSpMVOpDescr *newDescr, void *buffer)
{
    auto *hCtx = SpmvOpToInternalHandle(handle);
    int32_t mInt = static_cast<int32_t>(newDescr->m);
    uint32_t useBlocks = 0, rpb = 0;
    aclsparseStatus_t splitSt = ComputeSpmvOpBlockSplits(mInt, useBlocks, rpb);
    if (splitSt != ACL_SPARSE_STATUS_SUCCESS) {
        return splitSt;
    }
    newDescr->alg2NumBlocks = useBlocks;
    SpmvOpWorkspaceLayout layout{};
    layout.Compute(mInt, useBlocks);
    auto *wsBase = static_cast<uint8_t *>(buffer);
    spmv_op_preprocess_kernel_do(
        reinterpret_cast<GM_ADDR>(newDescr->csrRowOffsets),
        reinterpret_cast<GM_ADDR>(wsBase + layout.reorderOffset),
        reinterpret_cast<GM_ADDR>(wsBase + layout.tmpReorderOffset),
        reinterpret_cast<GM_ADDR>(wsBase + layout.scratchOffset),
        reinterpret_cast<GM_ADDR>(wsBase + layout.tmpScratchOffset),
        reinterpret_cast<GM_ADDR>(wsBase + layout.binEdgeOffset),
        mInt, useBlocks,
        (newDescr->rowOffsetType == ACL_SPARSE_INDEX_32I)
            ? SPMV_OP_IDX_RT_I32 : SPMV_OP_IDX_RT_I64,
        hCtx->stream);
    OP_LOGI(kTag, "createDescr ALG2: preprocess kernel launched async");
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 从 SpMatDescr 初始化 SpMVOpDescr 的所有字段（不含 alg/userBuffer/alg2NumBlocks）
static aclsparseStatus_t InitSpMVOpDescrFromSpMat(
    aclsparseSpMVOpDescr *newDescr, const aclsparseSpMatDescr *matInner)
{
    newDescr->format        = matInner->format;
    newDescr->m             = matInner->rows;
    newDescr->n             = matInner->cols;
    newDescr->nnz           = matInner->nnz;
    // 校验 indexBase 仅为 ZERO 或 ONE（异常枚举值会导致 kernel 索引偏移错误）
    if (matInner->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matInner->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(kTag, "createDescr: unsupported indexBase %d",
                static_cast<int>(matInner->baseType));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    newDescr->indexBase     = matInner->baseType;
    newDescr->rowOffsetType = matInner->ptrType;
    newDescr->colIndType    = matInner->IdxType;
    newDescr->valueType     = matInner->valueType;
    newDescr->csrRowOffsets = matInner->ptrs;
    newDescr->csrColInd     = matInner->idxs;
    newDescr->csrValues     = matInner->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F2: createDescr
aclsparseStatus_t aclsparseSpMVOp_createDescr(
    aclsparseHandle_t handle, aclsparseSpMVOpDescr_t *descr, aclsparseOperation_t opA, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY, aclsparseDnVecDescr_t vecZ,
    aclDataType computeType, aclsparseSpMVOpAlg_t alg, void *buffer)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "createDescr: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMVOpCreateDescrParams(descr, matA, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmvOpValidateCommon(kTag, opA, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    auto *matInner = SpmvOpToMatInner(matA);
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(kTag, "createDescr: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(matInner->rows));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    auto *rawDescr = new (std::nothrow) aclsparseSpMVOpDescr;
    if (rawDescr == nullptr) {
        OP_LOGE(kTag, "createDescr: alloc failed");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    std::unique_ptr<aclsparseSpMVOpDescr> newDescr(rawDescr);
    st = InitSpMVOpDescrFromSpMat(newDescr.get(), matInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    newDescr->alg = (alg == ACL_SPARSE_SPMVOP_ALG_DEFAULT) ? ACL_SPARSE_SPMVOP_ALG1 : alg;
    if (newDescr->alg == ACL_SPARSE_SPMVOP_ALG2 && buffer != nullptr) {
        st = CreateDescrPreprocessAlg2(handle, newDescr.get(), buffer);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    newDescr->userBuffer = (newDescr->alg == ACL_SPARSE_SPMVOP_ALG2) ? buffer : nullptr;
    OP_LOGI(kTag, "createDescr: m=%llu, nnz=%llu, alg=%d",
            static_cast<unsigned long long>(newDescr->m),
            static_cast<unsigned long long>(newDescr->nnz),
            static_cast<int>(newDescr->alg));
    *descr = reinterpret_cast<aclsparseSpMVOpDescr_t>(newDescr.release());
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F3: destroyDescr（幂等语义）
aclsparseStatus_t aclsparseSpMVOp_destroyDescr(aclsparseSpMVOpDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete reinterpret_cast<aclsparseSpMVOpDescr *>(descr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F4: createPlan
aclsparseStatus_t aclsparseSpMVOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMVOpDescr_t descr,
    aclsparseSpMVOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "createPlan: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMVOpCreatePlanParams(
        descr, plan, epilogueLTOBuffer, epilogueLTOBufferSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    auto *newPlan = new (std::nothrow) SpmvOpPlanData;
    if (newPlan == nullptr) {
        OP_LOGE(kTag, "createPlan: alloc failed");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    newPlan->descr = descr;
    newPlan->alg   = descr->alg;

    *plan = reinterpret_cast<aclsparseSpMVOpPlan_t>(newPlan);
    OP_LOGI(kTag, "createPlan: alg=%d", static_cast<int>(newPlan->alg));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F5: destroyPlan（幂等语义）
aclsparseStatus_t aclsparseSpMVOp_destroyPlan(aclsparseSpMVOpPlan_t plan)
{
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete reinterpret_cast<SpmvOpPlanData *>(plan);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F6: setGlobalUserData（identity epilogue，no-op）
aclsparseStatus_t aclsparseSpMVOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize)
{
    // ---- nullptr guard: handle & plan must be valid ----
    if (handle == nullptr) {
        OP_LOGE(kTag, "setGlobalUserData: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr) {
        OP_LOGE(kTag, "setGlobalUserData: plan is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 校验参数一致性：epilogueData 和 epilogueDataSize 必须同时有效或同时无效
    if ((epilogueData == nullptr) != (epilogueDataSize == 0)) {
        OP_LOGE(kTag, "setGlobalUserData: inconsistent params (data=%p, size=%zu)",
                epilogueData, epilogueDataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // NPU 侧使用 identity epilogue，无需任何数据；直接返回 SUCCESS
    return ACL_SPARSE_STATUS_SUCCESS;
}

// F7: 主执行入口
aclsparseStatus_t aclsparseSpMVOp(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ)
{
    if (handle == nullptr) {
        OP_LOGE(kTag, "execute: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // Cast opaque plan to internal struct
    auto *internalPlan = reinterpret_cast<SpmvOpPlanData *>(plan);
    aclsparseStatus_t st = ValidateSpMVOpExecuteParams(
        handle, internalPlan, alpha, beta, vecX, vecY, vecZ);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    return LaunchSpMVOpKernel(handle, internalPlan, alpha, beta, vecX, vecY, vecZ);
}

}  // extern "C"
