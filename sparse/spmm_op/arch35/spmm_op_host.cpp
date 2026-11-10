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
 * \file spmm_op_host.cpp
 * \brief spmm_op Host 侧实现：7 个 Generic API 入口。
 *
 * API 清单：
 *   F1. aclsparseSpMMOp_bufferSize
 *   F2. aclsparseSpMMOp_createDescr
 *   F3. aclsparseSpMMOp_destroyDescr
 *   F4. aclsparseSpMMOp_createPlan
 *   F5. aclsparseSpMMOp_destroyPlan
 *   F6. aclsparseSpMMOp_setGlobalUserData
 *   F7. aclsparseSpMMOp（主执行入口）
 *
 * 每个 API 内部拆分为 ValidateParams + LaunchKernel 两个 static 函数。
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_handle_internal.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_host_utils.h"
#include "spmm_op.h"
#include "spmm_op_tiling_data.h"
// Host 侧不引入 kernel_operator.h，此处提供 GM_ADDR 的 host 编译回退定义。
// NPU 侧由 toolkit (kernel_utils_macros.h) 自动定义为 __gm__ uint8_t*。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif
#include "spmm_op_kernel.h"

// Tag 常量，用于 dlog 的模块标识
static constexpr const char *SPMM_OP_TAG = "aclsparseSpMMOp";

namespace {

// Internal plan struct (opaque to external code; public type is void*)
// 生命周期约束：descr 必须先于 plan 销毁，即用户调用顺序必须为：
//   createDescr → createPlan → execute... → destroyPlan → destroyDescr
// 若违反此顺序（先 destroyDescr），plan->descr 将成为悬垂指针，行为未定义。
struct SpmmOpPlanData {
    aclsparseSpMMOpDescr *descr{nullptr};  // weak ref，descr 必须先于 plan 销毁
    aclsparseSpMMOpAlg_t alg{ACL_SPARSE_SPMMOP_ALG_DEFAULT};
};

// ===========================================================================
// 公共参数校验辅助函数
// ===========================================================================

// 校验 matA 枚举字段：CSR 格式、valueType∈{FP32,FP16}、
// ptrType∈{I32,I64}、IdxType=I32、baseType∈{ZERO,ONE}
static aclsparseStatus_t ValidateSpMatEnums(
    const char *api, const aclsparseSpMatDescr *inner)
{
    if (inner->format != ACL_SPARSE_FORMAT_CSR) {
        OP_LOGE(api, "unsupported format %d (CSR required)",
                static_cast<int>(inner->format));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->valueType != ACL_FLOAT && inner->valueType != ACL_FLOAT16) {
        OP_LOGE(api, "unsupported valueType %d (FP32/FP16 required)",
                static_cast<int>(inner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->IdxType != ACL_SPARSE_INDEX_32I) {
        OP_LOGE(api, "unsupported colIndType (IdxType) %d (I32 required)",
                static_cast<int>(inner->IdxType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->ptrType != ACL_SPARSE_INDEX_32I &&
        inner->ptrType != ACL_SPARSE_INDEX_64I) {
        OP_LOGE(api, "unsupported rowOffsetType %d (I32/I64 required)",
                static_cast<int>(inner->ptrType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (inner->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        inner->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(api, "unsupported indexBase %d (ZERO/ONE required)",
                static_cast<int>(inner->baseType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matA CSR device pointers 非空 + nnz INT32_MAX 上限
static aclsparseStatus_t ValidateSpMatPointers(
    const char *api, const aclsparseSpMatDescr *inner)
{
    // rowOffsets 始终需要 (m+1 个元素)，即使 nnz=0
    if (inner->ptrs == nullptr) {
        OP_LOGE(api, "csrRowOffsets is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // colInd/values 在 nnz > 0 时必须非空
    if (inner->nnz > 0) {
        if (inner->idxs == nullptr) {
            OP_LOGE(api, "csrColInd is nullptr (nnz=%llu > 0)",
                    static_cast<unsigned long long>(inner->nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (inner->values == nullptr) {
            OP_LOGE(api, "csrValues is nullptr (nnz=%llu > 0)",
                    static_cast<unsigned long long>(inner->nnz));
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    // nnz INT32_MAX 校验：kernel 内 CSR 循环索引为 int32_t，
    // nnz 超限会导致窄化溢出（UB）。
    if (inner->nnz > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(api, "nnz=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(inner->nnz));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matA 描述符：非空、CSR 格式、valueType∈{FP32,FP16}、
// ptrType∈{I32,I64}、IdxType=I32、baseType∈{ZERO,ONE}
static aclsparseStatus_t SpmmOpValidateSpMat(
    const char *api, aclsparseConstSpMatDescr_t matA)
{
    if (matA == nullptr) {
        OP_LOGE(api, "matA is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto *inner = SpmmOpToMatInner(matA);
    aclsparseStatus_t st = ValidateSpMatEnums(api, inner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    return ValidateSpMatPointers(api, inner);
}

// 校验 opA / opB / computeType（适用于 bufferSize/createDescr）
static aclsparseStatus_t SpmmOpValidateCommon(
    const char *api, aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclDataType computeType)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        OP_LOGE(api, "unsupported opA %d (NON_TRANSPOSE only)",
                static_cast<int>(opA));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE && opB != ACL_SPARSE_OP_TRANSPOSE) {
        OP_LOGE(api, "unsupported opB %d (NON_TRANSPOSE/TRANSPOSE only)",
                static_cast<int>(opB));
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
static aclsparseStatus_t SpmmOpValidateAlg(const char *api, aclsparseSpMMOpAlg_t alg)
{
    if (alg != ACL_SPARSE_SPMMOP_ALG_DEFAULT &&
        alg != ACL_SPARSE_SPMMOP_ALG1 &&
        alg != ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION &&
        alg != ACL_SPARSE_SPMMOP_ALG2) {
        OP_LOGE(api, "invalid alg %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// bufferSize 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMMOpBufferParams(
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpMMOpAlg_t alg, size_t *bufferSize)
{
    aclsparseStatus_t st = SpmmOpValidateCommon(SPMM_OP_TAG, opA, opB, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmmOpValidateSpMat(SPMM_OP_TAG, matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmmOpValidateAlg(SPMM_OP_TAG, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    if (matB == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "matB is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matC == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "matC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (bufferSize == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "bufferSize is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // INT32_MAX 校验：k/ldb/ldc 后续需存入 int32_t tiling 字段，超限会静默截断
    const auto *matInner = SpmmOpToMatInner(matA);
    if (matInner->cols > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "k=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(matInner->cols));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const auto *bInner = SpmmOpToDnMatInner(matB);
    if (bInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "matB.ld=%lld exceeds INT32_MAX, not supported",
                static_cast<long long>(bInner->ld));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const auto *cInner = SpmmOpToDnMatInner(matC);
    if (cInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "matC.ld=%lld exceeds INT32_MAX, not supported",
                static_cast<long long>(cInner->ld));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// createDescr 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMMOpCreateDescrParams(
    aclsparseSpMMOpDescr_t *descr, aclsparseConstSpMatDescr_t matA,
    aclsparseSpMMOpAlg_t alg, void *buffer)
{
    if (descr == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "descr output is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseStatus_t st = SpmmOpValidateSpMat(SPMM_OP_TAG, matA);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmmOpValidateAlg(SPMM_OP_TAG, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    // ALG2: buffer 必须非空（createDescr 预处理写入 reorder/bin_edge）
    // ALG1: buffer 可为 NULL
    if (alg == ACL_SPARSE_SPMMOP_ALG2 && buffer == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "buffer is nullptr (ALG2 requires buffer)");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// createPlan 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateSpMMOpCreatePlanParams(
    aclsparseSpMMOpDescr_t descr, aclsparseSpMMOpPlan_t *plan,
    const void *epilogueLTOBuffer, size_t epilogueLTOBufferSize)
{
    if (descr == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "descr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (plan == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "plan output is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // NPU 侧不支持 LTO-IR：若传入非 NULL 则返回 NOT_SUPPORTED
    if (epilogueLTOBuffer != nullptr || epilogueLTOBufferSize > 0) {
        OP_LOGE(SPMM_OP_TAG, "epilogue LTO not supported on NPU (buffer=%p, size=%zu)",
                epilogueLTOBuffer, epilogueLTOBufferSize);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// execute 参数校验
// ===========================================================================

static aclsparseStatus_t ValidateExecuteBasicParams(
    const SpmmOpPlanData *plan, const void *alpha, const void *beta)
{
    if (plan == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "plan is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (plan->descr == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "plan->descr is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "alpha is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "beta is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matB/matC dtype 一致性与维度
static aclsparseStatus_t ValidateDnMatsDtypeAndDims(
    const char *api, const aclsparseSpMMOpDescr *matInner,
    const aclsparseDnMatDescr *bInner, const aclsparseDnMatDescr *cInner)
{
    // dtype 一致性校验（A/B/C 必须一致）
    if (bInner->valueType != matInner->valueType ||
        cInner->valueType != matInner->valueType) {
        OP_LOGE(api, "dtype mismatch: A=%d, B=%d, C=%d (must be identical)",
                static_cast<int>(matInner->valueType),
                static_cast<int>(bInner->valueType),
                static_cast<int>(cInner->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // 维度校验
    const int64_t m = static_cast<int64_t>(matInner->m);
    const int64_t k = static_cast<int64_t>(matInner->k);
    const int64_t n = cInner->cols;
    if (cInner->rows != m) {
        OP_LOGE(api, "matC.rows=%ld != m=%ld", cInner->rows, m);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(api, "matC.cols=%ld < 0", n);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (n > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(api, "matC.cols=%ld exceeds INT32_MAX, not supported", n);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // opB 维度校验
    if (matInner->opB == ACL_SPARSE_OP_NON_TRANSPOSE) {
        // B 物理形状 (k×n)
        if (bInner->rows != k) {
            OP_LOGE(api, "matB.rows=%ld != k=%ld (opB=N)", bInner->rows, k);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (bInner->cols != n) {
            OP_LOGE(api, "matB.cols=%ld != n=%ld (opB=N)", bInner->cols, n);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    } else {
        // B 物理形状 (n×k)，op(B)=B^T[k×n]
        if (bInner->cols != k) {
            OP_LOGE(api, "matB.cols=%ld != k=%ld (opB=T)", bInner->cols, k);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        if (bInner->rows != n) {
            OP_LOGE(api, "matB.rows=%ld != n=%ld (opB=T)", bInner->rows, n);
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDnMatsOrder(
    const char *api, const aclsparseDnMatDescr *bInner,
    const aclsparseDnMatDescr *cInner)
{
    if (bInner->order != ACL_SPARSE_ORDER_ROW && bInner->order != ACL_SPARSE_ORDER_COL) {
        OP_LOGE(api, "matB.order=%d invalid (ROW/COL required)",
                static_cast<int>(bInner->order));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cInner->order != ACL_SPARSE_ORDER_ROW && cInner->order != ACL_SPARSE_ORDER_COL) {
        OP_LOGE(api, "matC.order=%d invalid (ROW/COL required)",
                static_cast<int>(cInner->order));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateDnMatsStrideLimit(
    const char *api, const aclsparseDnMatDescr *bInner,
    const aclsparseDnMatDescr *cInner)
{
    constexpr int64_t fp32StrideLimit = static_cast<int64_t>(UINT32_MAX) / 4;
    constexpr int64_t fp16StrideLimit = static_cast<int64_t>(UINT32_MAX) / 2;
    const int64_t strideLimit = (bInner->valueType == ACL_FLOAT) ? fp32StrideLimit : fp16StrideLimit;
    if (bInner->ld > strideLimit) {
        OP_LOGE(api, "matB.ld=%lld exceeds stride limit %lld for dtype, not supported",
                static_cast<long long>(bInner->ld), static_cast<long long>(strideLimit));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (cInner->ld > strideLimit) {
        OP_LOGE(api, "matC.ld=%lld exceeds stride limit %lld for dtype, not supported",
                static_cast<long long>(cInner->ld), static_cast<long long>(strideLimit));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matB/matC ld 与 values 指针
static aclsparseStatus_t ValidateDnMatsLdAndPointers(
    const char *api, const aclsparseDnMatDescr *bInner,
    const aclsparseDnMatDescr *cInner)
{
    aclsparseStatus_t st = ValidateDnMatsOrder(api, bInner, cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // ld 下限校验
    const bool bRowMajor = (bInner->order == ACL_SPARSE_ORDER_ROW);
    const bool cRowMajor = (cInner->order == ACL_SPARSE_ORDER_ROW);
    const int64_t bMinLd = bRowMajor ? bInner->cols : bInner->rows;
    const int64_t cMinLd = cRowMajor ? cInner->cols : cInner->rows;
    if (bInner->ld < bMinLd) {
        OP_LOGE(api, "matB.ld=%ld < minLd=%ld", bInner->ld, bMinLd);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cInner->ld < cMinLd) {
        OP_LOGE(api, "matC.ld=%ld < minLd=%ld", cInner->ld, cMinLd);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // ld 上限校验：INT32_MAX
    if (bInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(api, "matB.ld=%lld exceeds INT32_MAX, not supported",
                static_cast<long long>(bInner->ld));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (cInner->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(api, "matC.ld=%lld exceeds INT32_MAX, not supported",
                static_cast<long long>(cInner->ld));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    st = ValidateDnMatsStrideLimit(api, bInner, cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // values 非空校验
    if (bInner->values == nullptr) {
        OP_LOGE(api, "matB.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (cInner->values == nullptr) {
        OP_LOGE(api, "matC.values is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 matB/matC：非空、valueType 匹配、维度合法、order 已校验、ld 合法
static aclsparseStatus_t ValidateExecuteDnMats(
    const char *api,
    const aclsparseSpMMOpDescr *matInner,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC)
{
    if (matB == nullptr || matC == nullptr) {
        OP_LOGE(api, "matB/matC is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    const auto *bInner = SpmmOpToDnMatInner(matB);
    const auto *cInner = SpmmOpToDnMatInner(matC);
    aclsparseStatus_t st = ValidateDnMatsDtypeAndDims(api, matInner, bInner, cInner);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    return ValidateDnMatsLdAndPointers(api, bInner, cInner);
}

static aclsparseStatus_t ValidateSpMMOpExecuteParams(
    aclsparseHandle_t handle,
    const SpmmOpPlanData *plan, const void *alpha, const void *beta,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC)
{
    // handle 来自公共 API 签名，本函数内部不使用（仅透传至 launch 阶段）。
    (void)handle;
    aclsparseStatus_t st = ValidateExecuteBasicParams(plan, alpha, beta);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    return ValidateExecuteDnMats(SPMM_OP_TAG, plan->descr, matB, matC);
}

// ===========================================================================
// Tiling 计算（host 侧）
// ===========================================================================

static aclsparseStatus_t ComputeSpmmOpBlockSplits(
    int32_t m, uint32_t &useBlocks, uint32_t &rowsPerBlock)
{
    if (m <= 0) {
        useBlocks = 0;
        rowsPerBlock = 0;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
              OP_LOGE(SPMM_OP_TAG, "GetAivCoreCount returned 0");
              return ACL_SPARSE_STATUS_INTERNAL_ERROR);
    useBlocks = std::min(aivCoreNum,
        CeilDiv<uint32_t>(static_cast<uint32_t>(m), SPMM_OP_MAX_THREADS_PER_BLOCK));
    if (useBlocks == 0) {
        useBlocks = 1;
    }
    rowsPerBlock = CeilDiv<uint32_t>(static_cast<uint32_t>(m), useBlocks);
    if (rowsPerBlock == 0) {
        rowsPerBlock = 1;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static SpmmOpTilingData BuildSpmmOpTiling(
    const aclsparseSpMMOpDescr *descr,
    const aclsparseContext *h,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC,
    const void *alpha, const void *beta,
    uint32_t rowsPerBlock)
{
    SpmmOpTilingData tiling{};
    // 静态字段（descr，createDescr 绑定）
    tiling.m = static_cast<int32_t>(descr->m);
    tiling.indexBase = static_cast<int32_t>(descr->indexBase);  // ZERO 或 ONE
    tiling.algType = (descr->alg == ACL_SPARSE_SPMMOP_ALG2) ? SPMM_OP_ALG_TYPE_2 : SPMM_OP_ALG_TYPE_1;
    tiling.rowOffsetType = (descr->rowOffsetType == ACL_SPARSE_INDEX_64I) ? SPMM_OP_IDX_RT_I64 : SPMM_OP_IDX_RT_I32;
    tiling.opB = (descr->opB == ACL_SPARSE_OP_TRANSPOSE) ? SPMM_OP_OPB_TRANSPOSE : SPMM_OP_OPB_NON_TRANSPOSE;
    tiling.dtype = (descr->valueType == ACL_FLOAT16) ? SPMM_OP_DTYPE_FP16 : SPMM_OP_DTYPE_FP32;
    tiling.highPrecision = (descr->alg == ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION &&
                            descr->valueType == ACL_FLOAT) ? 1 : 0;
    // 动态字段（matB/matC 可跨 execute 变化）
    const auto *bInner = SpmmOpToDnMatInner(matB);
    const auto *cInner = SpmmOpToDnMatInner(matC);
    tiling.n = static_cast<int32_t>(cInner->cols);
    tiling.ldb = static_cast<int32_t>(bInner->ld);
    tiling.ldc = static_cast<int32_t>(cInner->ld);
    tiling.orderPair = static_cast<int32_t>(bInner->order) * 2 +
                       static_cast<int32_t>(cInner->order);
    // pointerMode: HOST mode 解引用 alpha/beta；DEVICE mode 存 alphaPtr/betaPtr
    if (h->pointerMode == ACL_SPARSE_POINTER_MODE_HOST) {
        tiling.alpha = (alpha != nullptr) ? *static_cast<const float *>(alpha) : 0.0f;
        tiling.beta  = (beta  != nullptr) ? *static_cast<const float *>(beta)  : 0.0f;
        tiling.alphaPtr = 0ULL;
        tiling.betaPtr  = 0ULL;
    } else {
        tiling.alphaPtr = reinterpret_cast<uint64_t>(alpha);
        tiling.betaPtr  = reinterpret_cast<uint64_t>(beta);
    }
    tiling.rowsPerBlock = rowsPerBlock;
    tiling.k = static_cast<int32_t>(descr->k);
    tiling.nTile = SPMM_OP_N_TILE;
    return tiling;
}

// ===========================================================================
// nnz=0 快捷路径：C = beta * C（NPU async kernel）
// ===========================================================================

static aclsparseStatus_t LaunchSpMMOpBetaCKernel(
    aclsparseHandle_t handle, const SpmmOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC)
{
    auto *h = SpmmOpToInternalHandle(handle);
    aclrtStream stream = h->stream;
    const auto *descr = plan->descr;
    if (descr->m > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "BetaC: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(descr->m));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(descr->m);

    uint32_t useBlocks = 0;
    uint32_t rowsPerBlock = 0;
    aclsparseStatus_t splitSt = ComputeSpmmOpBlockSplits(m, useBlocks, rowsPerBlock);
    CHECK_RET(splitSt == ACL_SPARSE_STATUS_SUCCESS, return splitSt);

    // 构造 tiling（仅 beta/m/n/ldc/orderPair/rowsPerBlock 有效）
    SpmmOpTilingData tiling = BuildSpmmOpTiling(descr, h, matB, matC, alpha, beta, rowsPerBlock);

    const auto *cInner = SpmmOpToDnMatInner(matC);

    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spmm_op_beta_c_kernel_do(
        static_cast<GM_ADDR>(cInner->values),
        tiling, useBlocks, stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE(SPMM_OP_TAG, "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }



    OP_LOGI(SPMM_OP_TAG, "beta_c kernel launched: m=%d, n=%d, numBlocks=%u", m, tiling.n, useBlocks);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ===========================================================================
// 主 kernel launch: C = alpha * op(A) * op(B) + beta * C（nnz > 0 时调用）
// ===========================================================================

static aclsparseStatus_t LaunchSpMMOpNnzKernel(
    aclsparseHandle_t handle, const SpmmOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC,
    int32_t m, uint64_t nnz)
{
    auto *h = SpmmOpToInternalHandle(handle);
    const auto *descr = plan->descr;
    uint32_t useBlocks = 0;
    uint32_t rowsPerBlock = 0;
    const bool isAlg2 = (descr->alg == ACL_SPARSE_SPMMOP_ALG2);
    if (isAlg2) {
        if (descr->alg2NumBlocks == 0) {
            OP_LOGE(SPMM_OP_TAG, "ALG2: alg2NumBlocks not set in descr");
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
        useBlocks = descr->alg2NumBlocks;
        rowsPerBlock = (static_cast<uint32_t>(m) + useBlocks - 1u) / useBlocks;
        if (rowsPerBlock == 0u) { rowsPerBlock = 1u; }
    } else {
        aclsparseStatus_t splitSt = ComputeSpmmOpBlockSplits(m, useBlocks, rowsPerBlock);
        CHECK_RET(splitSt == ACL_SPARSE_STATUS_SUCCESS, return splitSt);
    }
    SpmmOpTilingData tiling = BuildSpmmOpTiling(descr, h, matB, matC, alpha, beta, rowsPerBlock);
    const auto *bInner = SpmmOpToDnMatInner(matB);
    const auto *cInner = SpmmOpToDnMatInner(matC);
    auto *gmRowOffsets = static_cast<GM_ADDR>(descr->csrRowOffsets);
    auto *gmColInd = static_cast<GM_ADDR>(descr->csrColInd);
    auto *gmValues = static_cast<GM_ADDR>(descr->csrValues);
    GM_ADDR gmReorder = nullptr;
    GM_ADDR gmBinEdge = nullptr;
    if (tiling.algType == SPMM_OP_ALG_TYPE_2 && descr->userBuffer != nullptr) {
        auto *wsBase = static_cast<uint8_t *>(descr->userBuffer);
        SpmmOpWorkspaceLayout layout{};
        layout.Compute(m, useBlocks);
        gmReorder = wsBase + layout.reorderOffset;
        gmBinEdge = wsBase + layout.binEdgeOffset;
    }
    // clear-then-launch-then-peek（issue #357 分族）：Peek 只读不清除，
    // 下发前 aclrtGetLastError 清除陈旧线程级错误，只归因本次下发。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);

    spmm_op_kernel_do(
        gmRowOffsets, gmColInd, gmValues,
        static_cast<GM_ADDR>(bInner->values),
        static_cast<GM_ADDR>(cInner->values),
        gmReorder, gmBinEdge, tiling, useBlocks, h->stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE(SPMM_OP_TAG, "kernel enqueue failed: %d", static_cast<int>(aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }


    OP_LOGI(SPMM_OP_TAG, "spmm_op kernel launched: m=%d, n=%d, nnz=%llu, numBlocks=%u, alg=%d, dtype=%d, opB=%d",
            m, tiling.n, static_cast<unsigned long long>(nnz), useBlocks,
            static_cast<int>(tiling.algType), static_cast<int>(tiling.dtype),
            static_cast<int>(tiling.opB));
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t LaunchSpMMOpKernel(
    aclsparseHandle_t handle, const SpmmOpPlanData *plan,
    const void *alpha, const void *beta,
    aclsparseConstDnMatDescr_t matB, aclsparseConstDnMatDescr_t matC)
{
    const auto *descr = plan->descr;
    if (descr->m > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(descr->m));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(descr->m);
    if (m == 0) {
        OP_LOGD(SPMM_OP_TAG, "m=0, skip kernel launch");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    // n==0 时输出矩阵为空，无需计算
    const auto *cInner = SpmmOpToDnMatInner(matC);
    const int32_t n = static_cast<int32_t>(cInner->cols);
    if (n == 0) {
        OP_LOGD(SPMM_OP_TAG, "n=0, skip kernel launch (empty output)");
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (descr->nnz == 0) {
        return LaunchSpMMOpBetaCKernel(handle, plan, alpha, beta, matB, matC);
    }
    return LaunchSpMMOpNnzKernel(handle, plan, alpha, beta, matB, matC, m, descr->nnz);
}

}  // namespace

// ===========================================================================
// Public APIs
// ===========================================================================
extern "C" {
// 查询 workspace 大小（ALG1 返回 0，ALG2 返回 reorder+bin_edge 对齐量）
aclsparseStatus_t aclsparseSpMMOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    size_t *bufferSize)
{
    if (handle == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "bufferSize: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMMOpBufferParams(
        opA, opB, matA, matB, matC, computeType, alg, bufferSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    const auto *matInner = SpmmOpToMatInner(matA);

    // m > INT32_MAX 时截断，提前返回 NOT_SUPPORTED（避免 silent truncation）
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "bufferSize: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(matInner->rows));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int32_t m = static_cast<int32_t>(matInner->rows);

    if (alg == ACL_SPARSE_SPMMOP_ALG2) {
        // ALG2 需要 workspace：先估算 numBlocks 以计算 bin_edge 大小
        uint32_t useBlocks = 0;
        uint32_t rowsPerBlock = 0;
        aclsparseStatus_t splitSt = ComputeSpmmOpBlockSplits(m, useBlocks, rowsPerBlock);
        if (splitSt != ACL_SPARSE_STATUS_SUCCESS) { return splitSt; }
        SpmmOpWorkspaceLayout layout{};
        layout.Compute(m, useBlocks);
        *bufferSize = layout.totalBytes;
    } else {
        // ALG1 / DEFAULT: 无 workspace
        *bufferSize = 0;
    }

    OP_LOGD(SPMM_OP_TAG, "bufferSize: m=%d, alg=%d, bufferSize=%zu", m, static_cast<int>(alg), *bufferSize);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 读取 CSR rowOffsets 从 device 到 host，并计算 block splits 与 layout
static aclsparseStatus_t ReadAlg2HostRowOffsets(
    aclsparseSpMMOpDescr *newDescr, int32_t mInt, void *buffer,
    uint32_t &useBlocks, SpmmOpWorkspaceLayout &layout,
    std::vector<int64_t> &hostRowOffsets)
{
    (void)buffer;  // wsBase 在 WriteAlg2WorkspaceToDevice 中使用
    uint32_t rpb = 0;
    aclsparseStatus_t splitSt = ComputeSpmmOpBlockSplits(mInt, useBlocks, rpb);
    if (splitSt != ACL_SPARSE_STATUS_SUCCESS) {
        return splitSt;
    }
    newDescr->alg2NumBlocks = useBlocks;
    layout.Compute(mInt, useBlocks);

    // 读取 CSR rowOffsets 从 device 到 host（synchronous aclrtMemcpy）
    // 统一用 int64_t vector 存储，I32 路径先拷到临时 int32_t buffer 再提升
    aclError aclRet = ACL_ERROR_NONE;
    if (newDescr->rowOffsetType == ACL_SPARSE_INDEX_64I) {
        const size_t bytes = (static_cast<size_t>(mInt) + 1) * sizeof(int64_t);
        aclRet = aclrtMemcpy(hostRowOffsets.data(), bytes,
                             newDescr->csrRowOffsets, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            OP_LOGE(SPMM_OP_TAG, "ALG2 preprocess: aclrtMemcpy D2H rowOffsets(I64) failed: %d",
                    static_cast<int>(aclRet));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    } else {
        std::vector<int32_t> hostRowOffsets32(static_cast<size_t>(mInt) + 1);
        const size_t bytes = (static_cast<size_t>(mInt) + 1) * sizeof(int32_t);
        aclRet = aclrtMemcpy(hostRowOffsets32.data(), bytes,
                             newDescr->csrRowOffsets, bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            OP_LOGE(SPMM_OP_TAG, "ALG2 preprocess: aclrtMemcpy D2H rowOffsets(I32) failed: %d",
                    static_cast<int>(aclRet));
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
        for (size_t i = 0; i <= static_cast<size_t>(mInt); ++i) {
            hostRowOffsets[i] =
                static_cast<int64_t>(hostRowOffsets32[i]);
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// host-side 计算 reorder + bin_edge（与 device kernel 的 3 个 pass 等价）
// Pass1: rowNnz[i] = rowOffsets[i+1] - rowOffsets[i]; reorder[i] = i
// Pass2: 按 nnz 降序排序 (reorder, rowNnz)
// Pass3: 按排序后 nnz 累计计算 bin_edge[numBlocks+1]
static void ComputeAlg2ReorderAndBins(
    int32_t mInt, uint32_t useBlocks,
    const std::vector<int64_t> &hostRowOffsets,
    std::vector<int32_t> &hostReorder,
    std::vector<int32_t> &hostBinEdge)
{
    // Pass1 等价：计算每行 nnz + 初始化 reorder
    // rowNnz 用 int32_t（nnz ≤ INT32_MAX 保证不溢出），hostRowOffsets 为 int64_t
    std::vector<int32_t> rowNnz(static_cast<size_t>(mInt));
    hostReorder.resize(static_cast<size_t>(mInt));
    for (int32_t i = 0; i < mInt; ++i) {
        rowNnz[static_cast<size_t>(i)] = static_cast<int32_t>(
            hostRowOffsets[static_cast<size_t>(i + 1)] - hostRowOffsets[static_cast<size_t>(i)]);
        hostReorder[static_cast<size_t>(i)] = i;
    }

    // Pass2 等价：按 nnz 降序排序（stable_sort 保持行号顺序，与归并排序行为一致）
    std::stable_sort(hostReorder.begin(), hostReorder.end(),
                     [&rowNnz](int32_t a, int32_t b) {
                         return rowNnz[static_cast<size_t>(a)] > rowNnz[static_cast<size_t>(b)];
                     });

    // 构建排序后的 nnz 数组（对应 device kernel 中排序后的 scratch）
    std::vector<int32_t> sortedNnz(static_cast<size_t>(mInt));
    for (int32_t i = 0; i < mInt; ++i) {
        sortedNnz[static_cast<size_t>(i)] =
            rowNnz[static_cast<size_t>(hostReorder[static_cast<size_t>(i)])];
    }

    // Pass3 等价：按排序后 nnz 累计计算 bin_edge[numBlocks+1]
    hostBinEdge.resize(static_cast<size_t>(useBlocks) + 1);
    int64_t totalNnz = 0;
    for (int32_t i = 0; i < mInt; ++i) {
        totalNnz += sortedNnz[static_cast<size_t>(i)];
    }
    const int64_t nnzPerBlock = (totalNnz > 0 && useBlocks > 0)
        ? (totalNnz + static_cast<int64_t>(useBlocks) - 1) / static_cast<int64_t>(useBlocks)
        : 0;
    hostBinEdge[0] = 0;
    uint32_t binIdx = 1;
    int64_t runningNnz = 0;
    for (int32_t i = 0; i < mInt; ++i) {
        runningNnz += sortedNnz[static_cast<size_t>(i)];
        if (binIdx < useBlocks &&
            runningNnz >= nnzPerBlock * static_cast<int64_t>(binIdx)) {
            hostBinEdge[binIdx] = i + 1;
            binIdx++;
        }
    }
    while (binIdx <= useBlocks) {
        hostBinEdge[binIdx++] = mInt;
    }
}

// 写入 reorder 和 bin_edge 到 device workspace（synchronous aclrtMemcpy）
static aclsparseStatus_t WriteAlg2WorkspaceToDevice(
    uint8_t *wsBase, const SpmmOpWorkspaceLayout &layout,
    int32_t mInt, uint32_t useBlocks,
    const std::vector<int32_t> &hostReorder,
    const std::vector<int32_t> &hostBinEdge)
{
    aclError aclRet = aclrtMemcpy(wsBase + layout.reorderOffset,
                                  static_cast<size_t>(mInt) * sizeof(int32_t),
                                  hostReorder.data(),
                                  static_cast<size_t>(mInt) * sizeof(int32_t),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE(SPMM_OP_TAG, "ALG2 preprocess: aclrtMemcpy H2D reorder failed: %d",
                static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(wsBase + layout.binEdgeOffset,
                         static_cast<size_t>(useBlocks + 1) * sizeof(int32_t),
                         hostBinEdge.data(),
                         static_cast<size_t>(useBlocks + 1) * sizeof(int32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        OP_LOGE(SPMM_OP_TAG, "ALG2 preprocess: aclrtMemcpy H2D binEdge failed: %d",
                static_cast<int>(aclRet));
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// ALG2 预处理：host-side 计算 reorder + bin_edge，写入 device workspace
// ALG2 preprocess uses host-side computation (std::stable_sort + aclrtMemcpy)
// instead of device-side kernel, because dav-3510 AIV-only __aicore__ kernel
// directly accessing __gm__ is unstable for non-trivial matrix sizes.
// This approach has precedent in sparse/spmm/arch35/spmm_csr_mat.cpp.
static aclsparseStatus_t CreateDescrPreprocessAlg2(
    aclsparseSpMMOpDescr *newDescr, void *buffer)
{
    int32_t mInt = static_cast<int32_t>(newDescr->m);
    uint32_t useBlocks = 0;
    SpmmOpWorkspaceLayout layout{};
    std::vector<int64_t> hostRowOffsets(static_cast<size_t>(mInt) + 1);

    aclsparseStatus_t st = ReadAlg2HostRowOffsets(newDescr, mInt, buffer,
                                                  useBlocks, layout, hostRowOffsets);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    std::vector<int32_t> hostReorder;
    std::vector<int32_t> hostBinEdge;
    ComputeAlg2ReorderAndBins(mInt, useBlocks, hostRowOffsets, hostReorder, hostBinEdge);

    auto *wsBase = static_cast<uint8_t *>(buffer);
    st = WriteAlg2WorkspaceToDevice(wsBase, layout, mInt, useBlocks, hostReorder, hostBinEdge);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    OP_LOGI(SPMM_OP_TAG, "createDescr ALG2: host-side preprocess completed (m=%d, numBlocks=%u)",
            mInt, useBlocks);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 从 SpMatDescr 初始化 SpMMOpDescr 的所有字段（不含 alg/userBuffer/alg2NumBlocks）
static aclsparseStatus_t InitSpMMOpDescrFromSpMat(
    aclsparseSpMMOpDescr *newDescr, const aclsparseSpMatDescr *matInner,
    aclsparseOperation_t opB)
{
    newDescr->format     = matInner->format;
    newDescr->m          = matInner->rows;
    newDescr->k          = matInner->cols;
    newDescr->nnz        = matInner->nnz;
    newDescr->indexBase  = matInner->baseType;  // ZERO 或 ONE
    newDescr->rowOffsetType = matInner->ptrType;  // I32 或 I64
    newDescr->valueType  = matInner->valueType;
    newDescr->opB        = opB;  // createDescr 绑定，plan 生命周期固定
    // CSR device pointers（弱引用，不持有所有权）
    newDescr->csrRowOffsets = matInner->ptrs;
    newDescr->csrColInd     = matInner->idxs;
    newDescr->csrValues     = matInner->values;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 分配并初始化 SpMMOpDescr（不含 try/catch，由调用方包裹异常保护）
static aclsparseStatus_t CreateAndInitSpmmOpDescr(
    aclsparseSpMMOpDescr_t *descr, const aclsparseSpMatDescr *matInner,
    aclsparseOperation_t opB, aclsparseSpMMOpAlg_t alg, void *buffer)
{
    aclsparseSpMMOpDescr *raw = new (std::nothrow) aclsparseSpMMOpDescr;
    if (raw == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "createDescr: alloc failed");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    std::unique_ptr<aclsparseSpMMOpDescr> newDescr(raw);

    aclsparseStatus_t st = InitSpMMOpDescrFromSpMat(newDescr.get(), matInner, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // DEFAULT → ALG1 归一化；HIGH_PRECISION 保持原值不归一化
    newDescr->alg = (alg == ACL_SPARSE_SPMMOP_ALG_DEFAULT) ? ACL_SPARSE_SPMMOP_ALG1 : alg;
    if (newDescr->alg == ACL_SPARSE_SPMMOP_ALG2 && buffer != nullptr) {
        st = CreateDescrPreprocessAlg2(newDescr.get(), buffer);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }
    newDescr->userBuffer = (newDescr->alg == ACL_SPARSE_SPMMOP_ALG2) ? buffer : nullptr;
    *descr = static_cast<aclsparseSpMMOpDescr_t>(newDescr.release());
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 创建描述符，绑定 matA/opB/alg（ALG2 在此同步执行 host-side 预处理）
aclsparseStatus_t aclsparseSpMMOp_createDescr(
    aclsparseHandle_t handle, aclsparseSpMMOpDescr_t *descr,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpMMOpAlg_t alg, void *buffer)
{
    if (handle == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "createDescr: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMMOpCreateDescrParams(descr, matA, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    st = SpmmOpValidateCommon(SPMM_OP_TAG, opA, opB, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    const auto *matInner = SpmmOpToMatInner(matA);
    if (matInner->rows > static_cast<uint64_t>(INT32_MAX)) {
        OP_LOGE(SPMM_OP_TAG, "createDescr: m=%llu exceeds INT32_MAX, not supported",
                static_cast<unsigned long long>(matInner->rows));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // try/catch 保护 std::vector 分配穿越 extern "C" 边界（CreateDescrPreprocessAlg2
    // 内 6 处 vector 分配依赖默认抛异常 std::bad_alloc）。
    try {
        st = CreateAndInitSpmmOpDescr(descr, matInner, opB, alg, buffer);
        if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }
    } catch (const std::bad_alloc &e) {
        OP_LOGE(SPMM_OP_TAG, "createDescr: memory allocation failed: %s", e.what());
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    } catch (const std::exception &e) {
        OP_LOGE(SPMM_OP_TAG, "createDescr: exception: %s", e.what());
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    auto *newDescr = static_cast<aclsparseSpMMOpDescr *>(*descr);
    OP_LOGI(SPMM_OP_TAG, "createDescr: m=%llu, k=%llu, nnz=%llu, alg=%d, opB=%d",
            static_cast<unsigned long long>(newDescr->m),
            static_cast<unsigned long long>(newDescr->k),
            static_cast<unsigned long long>(newDescr->nnz),
            static_cast<int>(newDescr->alg),
            static_cast<int>(newDescr->opB));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 销毁描述符，释放 host 资源（幂等，nullptr 直接返回 SUCCESS）
aclsparseStatus_t aclsparseSpMMOp_destroyDescr(aclsparseSpMMOpDescr_t descr)
{
    if (descr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete static_cast<aclsparseSpMMOpDescr *>(descr);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 创建执行计划，绑定 descr（NPU 侧 identity epilogue，LTO 必须为 NULL）
aclsparseStatus_t aclsparseSpMMOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t descr,
    aclsparseSpMMOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize)
{
    if (handle == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "createPlan: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpMMOpCreatePlanParams(
        descr, plan, epilogueLTOBuffer, epilogueLTOBufferSize);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    auto *innerDescr = static_cast<aclsparseSpMMOpDescr *>(descr);
    auto raw = new (std::nothrow) SpmmOpPlanData;
    if (raw == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "createPlan: alloc failed");
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
    std::unique_ptr<SpmmOpPlanData> newPlan(raw);
    newPlan->descr = innerDescr;
    newPlan->alg   = innerDescr->alg;

    OP_LOGI(SPMM_OP_TAG, "createPlan: alg=%d", static_cast<int>(newPlan->alg));
    *plan = static_cast<aclsparseSpMMOpPlan_t>(static_cast<void *>(newPlan.release()));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 销毁执行计划（幂等，nullptr 直接返回 SUCCESS）
aclsparseStatus_t aclsparseSpMMOp_destroyPlan(aclsparseSpMMOpPlan_t plan)
{
    if (plan == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    delete static_cast<SpmmOpPlanData *>(static_cast<void *>(plan));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 设置 epilogue 全局数据（NPU 侧 identity epilogue no-op，直接返回 SUCCESS）
aclsparseStatus_t aclsparseSpMMOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize)
{
    // ---- nullptr guard: handle & plan must be valid ----
    if (handle == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "setGlobalUserData: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (plan == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "setGlobalUserData: plan is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // 校验参数一致性：epilogueData 和 epilogueDataSize 必须同时有效或同时无效
    if ((epilogueData == nullptr) != (epilogueDataSize == 0)) {
        OP_LOGE(SPMM_OP_TAG, "setGlobalUserData: inconsistent params (data=%p, size=%zu)",
                epilogueData, epilogueDataSize);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // NPU 侧使用 identity epilogue，无需任何数据；直接返回 SUCCESS
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 执行 C = alpha * op(A) * op(B) + beta * C（异步，matB/matC 可跨 execute 更换）
aclsparseStatus_t aclsparseSpMMOp(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC)
{
    if (handle == nullptr) {
        OP_LOGE(SPMM_OP_TAG, "execute: handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // Cast opaque plan to internal struct
    auto *internalPlan = static_cast<SpmmOpPlanData *>(static_cast<void *>(plan));
    aclsparseStatus_t st = ValidateSpMMOpExecuteParams(
        handle, internalPlan, alpha, beta, matB, matC);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    return LaunchSpMMOpKernel(handle, internalPlan, alpha, beta, matB, matC);
}

}  // extern "C"
