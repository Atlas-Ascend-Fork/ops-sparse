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

#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"
#include "spmm.h"
#include "spmm_csr_mat.h"
#include <iostream>
#include <limits>
#include <new>
#include <mutex>

static const uint32_t kFallbackBlockDim = 24;

// 按 pointer mode 安全读取标量：HOST 模式直接解引用 host 指针（nullptr 兜底 0），
// DEVICE 模式用同步拷贝从 device 取值，避免把 device 指针当 host 地址解引用导致崩溃。
static aclsparseStatus_t ReadScalarF32(const void *p, aclsparsePointerMode_t mode,
                                        float &out)
{
    if (p == nullptr) {
        out = 0.0f;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (mode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        aclError aclRet = aclrtMemcpy(&out, sizeof(float), p, sizeof(float),
                                      ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    } else {
        out = *static_cast<const float *>(p);
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

uint32_t GetSpmmBlockDim()
{
    static uint32_t cached = 0;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&]() {
        int64_t coreNum = 0;
        if (aclrtGetDeviceInfo(0, ACL_DEV_ATTR_AICORE_CORE_NUM, &coreNum) == ACL_ERROR_NONE &&
            coreNum > 0) {
            cached = static_cast<uint32_t>(coreNum);
        } else {
            cached = kFallbackBlockDim;
        }
    });
    return cached;
}

struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    uint64_t totalBytes;
};

WsOffsets ComputeWsOffsets(uint32_t M, uint32_t blockDim)
{
    WsOffsets off;
    off.tilingOff  = 64;
    off.reorderOff = off.tilingOff + (int64_t)ROUNDUP(sizeof(SpmmArch22TilingData), 64);
    off.binEdgeOff = off.reorderOff + (int64_t)ROUNDUP((uint64_t)M * sizeof(int32_t), 64);
    off.totalBytes = off.binEdgeOff + ROUNDUP(((uint64_t)blockDim + 1) * sizeof(int32_t), 64);
    return off;
}

// arch22 入参校验：公共部分（含 dtype 表驱动）见 aclsparse_descr_internal.h，此处仅保留 arch22 差异化检查（紧凑 ROW 布局、DEFAULT/CSR_ALG1）。

// arch22 仅支持全 FP32 dtype 组合
static const AclsparseSpmmDtypeCombo kSpmmDtypeCombos[] = {
    {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
};

// arch22 kernel 仅支持紧凑 ROW-major 布局（ld == cols）：kernel.cpp 按 B[col*N+j]、
// C[row*N+j] 硬编码寻址，未使用描述符的 order/ld。传入 COL-major 或带 padding 的
// ROW-major 会静默算错，故在此 fail-fast 拒绝。多布局需求请使用 arch35 实现。
static bool IsSupportedDnMatLayout(const aclsparseDnMatDescr *mat)
{
    return mat->order == ACL_SPARSE_ORDER_ROW &&
           mat->ld == mat->cols;
}

static bool IsSupportedSpmmAlg(aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_ALG_DEFAULT ||
           alg == ACL_SPARSE_SPMM_CSR_ALG1;
}

static aclsparseStatus_t ValidateSpmmInputs(const aclsparseSpMatDescr *matA,
                                            const aclsparseDnMatDescr *matB,
                                            const aclsparseDnMatDescr *matC,
                                            aclsparseOperation_t opA,
                                            aclsparseOperation_t opB,
                                            aclDataType computeType,
                                            aclsparseSpMMAlg_t alg)
{
    // 公共校验（非空/操作类型/格式/索引类型/base/dtype 组合/维度/零维）见
    // aclsparse_descr_internal.h；arch22 不支持 opB 转置、仅全 FP32
    aclsparseStatus_t st = AclsparseValidateSpmmCommon(
        matA, matB, matC, opA, opB, computeType, /*allowOpBTranspose=*/false,
        kSpmmDtypeCombos, sizeof(kSpmmDtypeCombos) / sizeof(kSpmmDtypeCombos[0]));
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    // arch22 kernel 仅支持紧凑 ROW-major（详见 IsSupportedDnMatLayout 注释）
    if (!IsSupportedDnMatLayout(matB) || !IsSupportedDnMatLayout(matC)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedSpmmAlg(alg)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // arch22 预处理使用 int32_t 存储行号和偏移，M/K/N/nnz 在后续处理中还会
    // 窄化为 uint32_t 类型的 Tiling 字段，因此在类型转换前拒绝不支持的形状。
    constexpr uint64_t kSpmmArch22DimensionUpperLimit =
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    if (matA->rows > kSpmmArch22DimensionUpperLimit ||
        matA->cols > kSpmmArch22DimensionUpperLimit ||
        static_cast<uint64_t>(matB->cols) > kSpmmArch22DimensionUpperLimit ||
        matA->nnz > kSpmmArch22DimensionUpperLimit) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMGetBufferSize(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size)
{
    auto *matAInner = (aclsparseSpMatDescr *)matA;
    auto *matBInner = (aclsparseDnMatDescr *)matB;
    auto *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    uint32_t blockDim = GetSpmmBlockDim();
    uint32_t M = static_cast<uint32_t>(matAInner->rows);
    WsOffsets off = ComputeWsOffsets(M, blockDim);
    *size = off.totalBytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t SpmmBuildTiling(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    const void *alpha, const void *beta,
    uint8_t *dBuffer, WsOffsets &off, aclsparsePointerMode_t pointerMode)
{
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint32_t blockDim = GetSpmmBlockDim();
    uint32_t M = static_cast<uint32_t>(matA->rows);
    uint32_t K = static_cast<uint32_t>(matA->cols);
    uint32_t N = static_cast<uint32_t>(matB->cols);
    uint32_t nnz = static_cast<uint32_t>(matA->nnz);

    SpmmArch22CsrMat csrMat(matA, blockDim);
    aclsparseStatus_t pre = csrMat.DoPreProcess(dBuffer, off.reorderOff, off.binEdgeOff);
    if (pre != ACL_SPARSE_STATUS_SUCCESS) {
        return pre;
    }

    float alphaHost = 0.0f;
    float betaHost  = 0.0f;
    aclsparseStatus_t sRet = ReadScalarF32(alpha, pointerMode, alphaHost);
    if (sRet != ACL_SPARSE_STATUS_SUCCESS) {
        return sRet;
    }
    sRet = ReadScalarF32(beta, pointerMode, betaHost);
    if (sRet != ACL_SPARSE_STATUS_SUCCESS) {
        return sRet;
    }

    SpmmArch22TilingData td{};
    td.M = M;
    td.K = K;
    td.N = N;
    td.nnz = nnz;
    td.blockDim = blockDim;
    td.nChunks = SpmmArch22ComputeNChunks(N);
    td.alphaHost = alphaHost;
    td.betaHost  = betaHost;
    td.reorderOffset = off.reorderOff;
    td.binEdgeOffset = off.binEdgeOff;

    aclError aclRet = aclrtMemcpy(dBuffer + off.tilingOff, sizeof(td),
                                  &td, sizeof(td), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMPreprocess(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    auto *matAInner = (aclsparseSpMatDescr *)matA;
    auto *matBInner = (const aclsparseDnMatDescr *)matB;
    auto *matCInner = (aclsparseDnMatDescr *)matC;
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    uint32_t blockDim = GetSpmmBlockDim();
    uint32_t M = static_cast<uint32_t>(matAInner->rows);
    WsOffsets off = ComputeWsOffsets(M, blockDim);

    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t pmSt = aclsparseGetPointerMode(handle, &pointerMode);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        return pmSt;
    }
    st = SpmmBuildTiling(matAInner, matBInner, matCInner,
                         alpha, beta, (uint8_t *)buffer, off, pointerMode);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    matAInner->activeBuffer = buffer;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMM(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclrtStream stream = nullptr;
    aclsparseStatus_t st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    auto *matAInner = (aclsparseSpMatDescr *)matA;
    auto *matBInner = (const aclsparseDnMatDescr *)matB;
    auto *matCInner = (aclsparseDnMatDescr *)matC;
    st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                            opA, opB, computeType, alg);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    uint32_t blockDim = GetSpmmBlockDim();
    uint32_t M = static_cast<uint32_t>(matAInner->rows);
    WsOffsets off = ComputeWsOffsets(M, blockDim);

    if (matAInner->activeBuffer != buffer) {
        aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
        aclsparseStatus_t pmSt = aclsparseGetPointerMode(handle, &pointerMode);
        if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
            return pmSt;
        }
        st = SpmmBuildTiling(matAInner, matBInner, matCInner,
                             alpha, beta, (uint8_t *)buffer, off, pointerMode);
        if (st != ACL_SPARSE_STATUS_SUCCESS) {
            return st;
        }
    }

    spmm_arch22_kernel_launch(
        reinterpret_cast<GM_ADDR>(matAInner->ptrs),
        reinterpret_cast<GM_ADDR>(matAInner->idxs),
        reinterpret_cast<GM_ADDR>(matAInner->values),
        reinterpret_cast<GM_ADDR>(matBInner->values),
        reinterpret_cast<GM_ADDR>(matCInner->values),
        reinterpret_cast<GM_ADDR>(buffer),
        reinterpret_cast<GM_ADDR>(buffer) + off.tilingOff,
        blockDim, stream);

    return ACL_SPARSE_STATUS_SUCCESS;
}
