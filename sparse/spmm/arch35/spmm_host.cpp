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

#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <new>
#include <mutex>

#include "log/log.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "log/log.h"
#include "spmm.h"
#include "spmm_csr_mat.h"

#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

extern "C" void spmm_kernel_launch(
    GM_ADDR csrRowOffsets, GM_ADDR csrColInd,
    GM_ADDR csrValues, GM_ADDR matB, GM_ADDR matC,
    GM_ADDR workspaceGM, GM_ADDR tilingGM,
    int32_t dataType, uint32_t blockDim, void *stream);

namespace {

// Outer launch blockDim = AIV core count from CANN platform config
// (e.g. Ascend950PR_9589: vector_core_cnt=64). Falls back to 24 (A2 default)
// if PlatformAscendC is unavailable.
constexpr uint32_t kSpmmBlockDimFallback = 24u;

uint32_t GetSpmmBlockDim()
{
    static uint32_t cached = 0u;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
#ifndef __CCE_AICORE__
        auto *plat = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (plat != nullptr) {
            const uint32_t aiv = plat->GetCoreNumAiv();
            if (aiv > 0u) {
                cached = aiv;
                return;
            }
        }
#endif
        cached = kSpmmBlockDimFallback;
    });
    return cached;
}

float ScalarReadF32(const void *p) {
    if (p == nullptr) {
        return 0.0f;
    }
    return *static_cast<const float *>(p);
}

int32_t ScalarReadI32(const void *p) {
    if (p == nullptr) {
        return 0;
    }
    return *static_cast<const int32_t *>(p);
}

void SetTilingAlphaBeta(SpmmTilingData *td,
    const void *alpha, const void *beta, aclDataType computeType)
{
    if (computeType == ACL_INT32) {
        td->alpha_host = static_cast<float>(ScalarReadI32(alpha));
        td->beta_host = static_cast<float>(ScalarReadI32(beta));
    } else {
        td->alpha_host = ScalarReadF32(alpha);
        td->beta_host = ScalarReadF32(beta);
    }
}

int32_t ComputeSpmmHighPrecision(aclsparseSpMMAlg_t alg, int32_t dataType)
{
    return (alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
            dataType == SPMM_DTYPE_FP32) ?
               1 :
               0;
}

// arch35 支持的 dtype 组合：统一 FP32 / FP16 / INT8 量化
static const AclsparseSpmmDtypeCombo kSpmmDtypeCombos[] = {
    {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT},
    {ACL_INT8, ACL_INT8, ACL_INT32, ACL_INT32},
};

static bool IsSupportedSpmmAlg(aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPMM_CSR_ALG1 ||
           alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG;
}

aclsparseStatus_t ValidateSpmmInputs(const aclsparseSpMatDescr *matA,
                                   const aclsparseDnMatDescr *matB,
                                   const aclsparseDnMatDescr *matC,
                                   aclsparseOperation_t opA,
                                   aclsparseOperation_t opB,
                                   aclDataType computeType,
                                   aclsparseSpMMAlg_t alg)
{
    // 公共校验（非空/操作类型/格式/索引类型/base/dtype 组合/维度匹配/零维）见
    // aclsparse_descr_internal.h；arch35 支持 opB 转置
    aclsparseStatus_t st = AclsparseValidateSpmmCommon(
        matA, matB, matC, opA, opB, computeType, /*allowOpBTranspose=*/true,
        kSpmmDtypeCombos, sizeof(kSpmmDtypeCombos) / sizeof(kSpmmDtypeCombos[0]));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMM", "ValidateSpmmInputs common check failed, status=%d",
                static_cast<int>(st));
        return st;
    }
    // dense ld 上限校验：ld 需存入 int32_t tiling 字段，超限静默截断风险（上游 d86aa93 语义保留）
    if (matB->ld > static_cast<int64_t>(INT32_MAX) ||
        matC->ld > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSpMM",
                "dense leading dimension exceeds INT32_MAX: matB->ld=%ld, matC->ld=%ld",
                matB->ld, matC->ld);
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedSpmmAlg(alg)) {
        OP_LOGE("aclsparseSpMM", "unsupported SpMM algorithm %d", static_cast<int>(alg));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->rows > static_cast<uint64_t>(INT32_MAX) ||
        matC->cols > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE("aclsparseSpMM", "matrix dimension exceeds INT32_MAX: matA->rows=%llu, matC->cols=%lld",
                static_cast<unsigned long long>(matA->rows), static_cast<long long>(matC->cols));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Returns workspace start byte offsets for the TilingData / reorder / bin-edge
// regions, computed once and shared by GetBufferSize / Preprocess / Spmm.
struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t totalBytes;
};

WsOffsets ComputeWsOffsets(int64_t m, int32_t blockDim) {
    WsOffsets o;
    o.tilingOff  = SPMM_WS_HEADER_BYTES;
    o.reorderOff = spmm_align_up(o.tilingOff + static_cast<int64_t>(sizeof(SpmmTilingData)), SPMM_WS_ALIGN);
    o.binEdgeOff = spmm_align_up(o.reorderOff + static_cast<int64_t>(sizeof(int32_t)) * m, SPMM_WS_ALIGN);
    o.totalBytes = spmm_align_up(o.binEdgeOff + static_cast<int64_t>(sizeof(int32_t)) * (blockDim + 1), SPMM_WS_ALIGN);
    return o;
}

} // namespace

aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMMGetBufferSize", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (size == nullptr) {
        OP_LOGE("aclsparseSpMMGetBufferSize", "size is nullptr");
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMGetBufferSize",
                "ValidateSpmmInputs failed, status=%d", static_cast<int>(st));
        return st;
    }
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matAInner->rows),
                                     static_cast<int32_t>(GetSpmmBlockDim()));
    *size = static_cast<size_t>(off.totalBytes);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 计算 SpMM 的 reorder/bin/tiling 并写入 device workspace buffer。
// 该函数只读 matA，不修改其内部状态（buffer 是否 active 由调用方决定）。
static aclsparseStatus_t SpmmBuildTilingToBuffer(
    aclsparseSpMatDescr *matAInner, aclsparseDnMatDescr *matBInner,
    aclsparseDnMatDescr *matCInner, aclsparseOperation_t opB,
    const void *alpha, const void *beta, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    const int64_t m = static_cast<int64_t>(matAInner->rows);
    WsOffsets off = ComputeWsOffsets(m, static_cast<int32_t>(GetSpmmBlockDim()));

    // 1) Compute reorder table + bin edges, copy into device workspace.
    SpmmCsrMat csr(matAInner, static_cast<int32_t>(GetSpmmBlockDim()));
    uint32_t pre = csr.DoPreProcess(static_cast<uint8_t *>(buffer),
                                    off.reorderOff,
                                    off.binEdgeOff,
                                    /*identity=*/false);
    if (pre != ACL_SPARSE_STATUS_SUCCESS) {
        return static_cast<aclsparseStatus_t>(pre);
    }

    // 2) Build TilingData on host, write to device workspace.
    SpmmTilingData td{};
    td.m   = static_cast<int32_t>(matAInner->rows);
    td.n   = static_cast<int32_t>(matCInner->cols);
    td.ldb = static_cast<int32_t>(matBInner->ld);
    td.ldc = static_cast<int32_t>(matCInner->ld);
    td.reorder_offset  = static_cast<int32_t>(off.reorderOff);
    td.bin_edge_offset = static_cast<int32_t>(off.binEdgeOff);
    td.high_precision  = ComputeSpmmHighPrecision(alg, SpmmDataTypeFromAcl(matAInner->valueType));
    td.opB             = (opB == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.order_pair      = static_cast<int32_t>(matBInner->order) * 2 +
                         static_cast<int32_t>(matCInner->order);
    SetTilingAlphaBeta(&td, alpha, beta, computeType);

    aclError aclRet = aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
                                  sizeof(SpmmTilingData),
                                  &td, sizeof(SpmmTilingData),
                                  ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmRefreshActiveTiling(
    void *buffer, const WsOffsets &off,
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg)
{
    SpmmTilingData td;
    aclError ret = aclrtMemcpy(&td, sizeof(SpmmTilingData),
                               static_cast<uint8_t *>(buffer) + off.tilingOff,
                               sizeof(SpmmTilingData),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    td.opB = (opB == ACL_SPARSE_OP_TRANSPOSE) ? 1 : 0;
    td.order_pair = static_cast<int32_t>(matB->order) * 2 +
                    static_cast<int32_t>(matC->order);
    td.high_precision = ComputeSpmmHighPrecision(alg, SpmmDataTypeFromAcl(matA->valueType));
    SetTilingAlphaBeta(&td, alpha, beta, computeType);
    ret = aclrtMemcpy(static_cast<uint8_t *>(buffer) + off.tilingOff,
                      sizeof(SpmmTilingData),
                      &td, sizeof(SpmmTilingData),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmEnsureTilingReady(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opB, const void *alpha, const void *beta,
    aclDataType computeType, aclsparseSpMMAlg_t alg, void *buffer, const WsOffsets &off)
{
    if (matA->activeBuffer == buffer) {
        return SpmmRefreshActiveTiling(buffer, off, matA, matB, matC,
                                       opB, alpha, beta, computeType, alg);
    }
    return SpmmBuildTilingToBuffer(matA, matB, matC, opB,
                                   alpha, beta, computeType, alg, buffer);
}

static aclsparseStatus_t SpmmRunKernel(
    aclsparseSpMatDescr *matA, aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    void *buffer, const WsOffsets &off, aclrtStream stream)
{
    spmm_kernel_launch(reinterpret_cast<GM_ADDR>(matA->ptrs),
                       reinterpret_cast<GM_ADDR>(matA->idxs),
                       reinterpret_cast<GM_ADDR>(matA->values),
                       reinterpret_cast<GM_ADDR>(matB->values),
                       reinterpret_cast<GM_ADDR>(matC->values),
                       reinterpret_cast<GM_ADDR>(buffer),
                       reinterpret_cast<GM_ADDR>(
                           static_cast<uint8_t *>(buffer) + off.tilingOff),
                       SpmmDataTypeFromAcl(matA->valueType),
                       GetSpmmBlockDim(), stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    if (handle == nullptr) {
        OP_LOGE("aclsparseSpMMPreprocess", "handle is nullptr");
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMPreprocess",
                "ValidateSpmmInputs failed, status=%d", static_cast<int>(st));
        return st;
    }
    if (buffer == nullptr) {
        OP_LOGE("aclsparseSpMMPreprocess", "buffer is nullptr");
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    st = SpmmBuildTilingToBuffer(matAInner, matBInner, matCInner, opB,
                                 alpha, beta, computeType, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseSpMMPreprocess",
                "SpmmBuildTilingToBuffer failed, status=%d", static_cast<int>(st));
        return st;
    }

    // 让这块 buffer 成为 matA 的 active buffer（Preprocess 写 matA 内部状态）。
    matAInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclsparseSpMatDescr *matAInner = (aclsparseSpMatDescr *)matA;
    aclsparseDnMatDescr *matBInner = (aclsparseDnMatDescr *)matB;
    aclsparseDnMatDescr *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (buffer == nullptr) {
        return ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclrtStream stream = nullptr;
    aclsparseGetStream(handle, &stream);
    WsOffsets off = ComputeWsOffsets(static_cast<int64_t>(matAInner->rows),
                                     static_cast<int32_t>(GetSpmmBlockDim()));
    st = SpmmEnsureTilingReady(matAInner, matBInner, matCInner, opB,
                               alpha, beta, computeType, alg, buffer, off);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return SpmmRunKernel(matAInner, matBInner, matCInner, buffer, off, stream);
}
