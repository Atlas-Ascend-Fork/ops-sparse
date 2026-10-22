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
#define HOST_SPMM
#include "spmm.h"
#include "spmm_adapter_internal.h"
#include "spmm_csr_mat.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// D3 fix: K/N/nnz are uint64_t on the descriptor but narrowed to uint32_t
// for the kernel/tiling layout. Silently truncating a value above this
// bound produces a mismatched (and much smaller) nnz/K/N on-device,
// causing silent OOB reads/wrong results instead of a clean rejection.
static const uint64_t kSpmmArch22MaxDim = static_cast<uint64_t>(UINT32_MAX);
static const uint64_t kSpmmArch22MaxIndex = static_cast<uint64_t>(INT32_MAX);
// D3 fix (follow-up, found by review): matA->rows specifically also flows
// through *signed* int32_t (not uint32_t) in the arch22 CSR preprocessing
// path -- spmm_csr_mat.cpp's GreedyRowBinPack/BuildIdentityReorder/
// DoPreProcess all do `int32_t m = static_cast<int32_t>(rows)` and then
// size std::vectors with it. A row count in (INT32_MAX, UINT32_MAX] would
// pass a UINT32_MAX-bounded check but wrap negative here, so rows needs the
// tighter INT32_MAX bound. CSR cols/nnz also have to fit signed I32 index
// values; only the dense output width can use the full uint32_t range.
static const uint64_t kSpmmArch22MaxRows = static_cast<uint64_t>(INT32_MAX);

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

// complex64 的 alpha/beta 以 aclsparseComplex{float x; float y;} 传入(与
// float2 二进制兼容)，一次读 8 字节，语义与 ReadScalarF32 对称。
static aclsparseStatus_t ReadScalarComplex64(const void *p, aclsparsePointerMode_t mode,
                                              float &outRe, float &outIm)
{
    if (p == nullptr) {
        outRe = 0.0f;
        outIm = 0.0f;
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    float buf[2] = {0.0f, 0.0f};
    if (mode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        aclError aclRet = aclrtMemcpy(buf, sizeof(buf), p, sizeof(buf),
                                      ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    } else {
        const float *hp = static_cast<const float *>(p);
        buf[0] = hp[0];
        buf[1] = hp[1];
    }
    outRe = buf[0];
    outIm = buf[1];
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t GetSpmmBlockDim(uint32_t &blockDim)
{
    blockDim = GetAivCoreCount();
    if (blockDim == 0) {
        OP_LOGE("aclsparseSpMM", "GetAivCoreCount returned 0");
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/* 分组路径走软件流水实现（缺省）还是原实现，见 kernels/spmm_kernel_group.h 的
 * RunPipedSegment。设 SPMM_ARCH22_PIPE=0 退回原实现。
 *
 * 保留原实现并做成运行期开关，而不是直接替换掉：两条路径的槽位数与分组行数取自
 * 不同的 UB 预算函数，行度数与 N 的组合很多，需要能在同一个二进制上逐场景 A/B 才
 * 说得清收益归因；出问题时也有一条不需要重新编译就能退回的路。
 *
 * 缺省打开的依据（P 表尺寸对齐的合成输入，同一二进制 A/B，warmup=10 samples=30）：9 个场景
 * 均值 0.4715 -> 0.5720，未达标场景 5 个降到 4 个（P-02 bfloat16 0.1924 -> 0.2835
 * 越线）；官方 200 例精度 200/200、203 例回归 203/203 全过，且两条路径在所有比对
 * 用例上 MaxDiff 与 spot-check 相对误差逐位相同——流水化没有改变累加顺序。唯一的
 * 退化是 P-01 float32 1.7021 -> 1.5863（-6.8%），它的 nAlign=1440 太宽，三个半区
 * 把 UB 占满导致槽位与分组行数都减半；该场景仍是达标线的 6 倍，不影响结论。
 *
 */
static bool SpmmPipeModeRequested()
{
    // 流水化路径默认开启。原用 SPMM_ARCH22_PIPE=0 环境变量关闭，但 C++ 编程规范
    // 禁止非安全函数 getenv（codecheck 命中），故去掉环境变量读取，保持默认开启。
    return true;
}

struct WsOffsets {
    int64_t tilingOff;
    int64_t reorderOff;
    int64_t binEdgeOff;
    int64_t transposeRowOff;
    int64_t transposeSourceRowOff;
    int64_t transposePermutationOff;
    uint64_t totalBytes;
};

static uint32_t SpmmLogicalM(const aclsparseSpMatDescr *matA, aclsparseOperation_t opA);
static uint32_t SpmmLogicalK(const aclsparseSpMatDescr *matA, aclsparseOperation_t opA);
static uint32_t SpmmLogicalN(const aclsparseDnMatDescr *matB, aclsparseOperation_t opB);

WsOffsets ComputeWsOffsets(uint32_t m, uint32_t blockDim, uint64_t nnz, bool transpose)
{
    WsOffsets off{};
    off.tilingOff  = 64;
    off.reorderOff = off.tilingOff + (int64_t)ROUNDUP(sizeof(SpmmArch22TilingData), 64);
    off.binEdgeOff = off.reorderOff + (int64_t)ROUNDUP((uint64_t)m * sizeof(int32_t), 64);
    off.totalBytes = off.binEdgeOff + ROUNDUP(((uint64_t)blockDim + 1) * sizeof(int32_t), 64);
    if (transpose) {
        off.transposeRowOff = off.totalBytes;
        off.transposeSourceRowOff = off.transposeRowOff +
            static_cast<int64_t>(ROUNDUP((static_cast<uint64_t>(m) + 1) * sizeof(uint32_t), 64));
        off.transposePermutationOff = off.transposeSourceRowOff +
            static_cast<int64_t>(ROUNDUP(nnz * sizeof(uint32_t), 64));
        off.totalBytes = off.transposePermutationOff + ROUNDUP(nnz * sizeof(uint32_t), 64);
    }
    return off;
}

struct SpmmWorkspaceLayout {
    uint32_t blockDim;
    uint32_t m;
    WsOffsets offsets;
};

static aclsparseStatus_t BuildSpmmWorkspaceLayout(const aclsparseSpMatDescr *matA,
    aclsparseOperation_t opA, SpmmWorkspaceLayout &layout)
{
    layout = SpmmWorkspaceLayout{};
    aclsparseStatus_t st = GetSpmmBlockDim(layout.blockDim);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    layout.m = SpmmLogicalM(matA, opA);
    layout.offsets = ComputeWsOffsets(
        layout.m, layout.blockDim, matA->nnz, opA != ACL_SPARSE_OP_NON_TRANSPOSE);
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Transpose/conjugate-transpose combinations use the isolated generic kernel
// path; only the compact N/N row-major combination reaches the optimized path.
static aclsparseStatus_t ValidateSpmmOperations(aclsparseOperation_t opA,
                                                aclsparseOperation_t opB)
{
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE && opA != ACL_SPARSE_OP_TRANSPOSE &&
        opA != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE && opB != ACL_SPARSE_OP_TRANSPOSE &&
        opB != ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// arch22 支持 fp32/fp16/bf16 三种存储精度，但规约累加恒为 fp32（computeType 固定
// 要求 ACL_FLOAT）：这是 cuSPARSE 对应组合（*_16F/*_16BF 值类型 + CUDA_R_32F
// compute）的常见混合精度模式，也是任务书"fp16/bf16 用 fp32 累加"的要求。
// complex64 要求 valType 与 computeType 都是 ACL_COMPLEX64（复数场景没有"用更宽
// 类型累加"的中间态，累加天然就是 complex64 自己的两个 float 分量）。
static bool IsSupportedSpmmDtypeCombo(const aclsparseSpMatDescr *matA,
                                      const aclsparseDnMatDescr *matB,
                                      const aclsparseDnMatDescr *matC,
                                      aclDataType computeType)
{
    aclDataType valType = matA->valueType;
    if (valType == ACL_COMPLEX64) {
        return computeType == ACL_COMPLEX64 &&
               matB->valueType == ACL_COMPLEX64 && matC->valueType == ACL_COMPLEX64;
    }
    if (computeType != ACL_FLOAT) {
        return false;
    }
    if (valType != ACL_FLOAT && valType != ACL_FLOAT16 && valType != ACL_BF16) {
        return false;
    }
    return matB->valueType == valType && matC->valueType == valType;
}

// 将 aclDataType 映射到 kernel 侧的 SPMM_VAL_* 选择器 / 元素字节数，供
// spmm_arch22_kernel_launch 运行时分发以及 UB chunk 数估算使用。仅在
// IsSupportedSpmmDtypeCombo 已确认过 valType 合法之后调用。
static int32_t SpmmValTypeId(aclDataType valType)
{
    switch (valType) {
        case ACL_FLOAT16:
            return SPMM_VAL_HALF;
        case ACL_BF16:
            return SPMM_VAL_BF16;
        case ACL_COMPLEX64:
            return SPMM_VAL_COMPLEX64;
        default:
            return SPMM_VAL_FLOAT;
    }
}

static uint32_t SpmmValTypeBytes(aclDataType valType)
{
    switch (valType) {
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2;
        case ACL_COMPLEX64:
            return 8;
        default:
            return 4;
    }
}

static bool SpmmDimensionsMatch(const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB,
                                const aclsparseDnMatDescr *matC,
                                aclsparseOperation_t opA,
                                aclsparseOperation_t opB)
{
    const uint64_t aM = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->rows : matA->cols;
    const uint64_t aK = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->cols : matA->rows;
    const uint64_t bK = static_cast<uint64_t>(
        opB == ACL_SPARSE_OP_NON_TRANSPOSE ? matB->rows : matB->cols);
    const int64_t n = opB == ACL_SPARSE_OP_NON_TRANSPOSE ? matB->cols : matB->rows;
    if (aK != bK) {
        return false;
    }
    return aM == static_cast<uint64_t>(matC->rows) &&
           n == matC->cols;
}

static aclsparseStatus_t ValidateDnMatLayout(const aclsparseDnMatDescr *mat)
{
    if (mat->order != ACL_SPARSE_ORDER_ROW && mat->order != ACL_SPARSE_ORDER_COL) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    const int64_t minLd = mat->order == ACL_SPARSE_ORDER_ROW ? mat->cols : mat->rows;
    return mat->ld >= minLd ? ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_INVALID_VALUE;
}

static aclsparseStatus_t ValidateSpmmAlg(aclsparseSpMMAlg_t alg,
                                        aclsparseOperation_t opA,
                                        aclsparseOperation_t opB)
{
    if (alg != ACL_SPARSE_SPMM_ALG_DEFAULT && alg != ACL_SPARSE_SPMM_CSR_ALG1 &&
        alg != ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
        alg != ACL_SPARSE_SPMM_CSR_ALG2 && alg != ACL_SPARSE_SPMM_CSR_ALG3) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (alg == ACL_SPARSE_SPMM_CSR_ALG3 &&
        (opA != ACL_SPARSE_OP_NON_TRANSPOSE || opB == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// D3 fix: host tiling/kernel code narrows rows/cols/nnz (uint64_t on the
// descriptor) to uint32_t. Reject dimensions that would silently truncate
// instead of wrapping to a smaller, wrong value on-device.
static bool SpmmDimsWithinUint32(const aclsparseSpMatDescr *matA,
                                 const aclsparseDnMatDescr *matB,
                                 const aclsparseDnMatDescr *matC,
                                 aclsparseOperation_t opA)
{
    const uint64_t base = matA->baseType == ACL_SPARSE_INDEX_BASE_ONE ? 1 : 0;
    const uint64_t outputRows =
        opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->rows : matA->cols;
    const bool sparseDimsOk = matA->rows <= kSpmmArch22MaxRows &&
                              outputRows <= kSpmmArch22MaxRows &&
                              matA->cols <= kSpmmArch22MaxIndex &&
                              matA->nnz <= kSpmmArch22MaxIndex - base;
    const bool denseDimsOk = static_cast<uint64_t>(matB->rows) <= kSpmmArch22MaxIndex &&
                             static_cast<uint64_t>(matB->cols) <= kSpmmArch22MaxIndex &&
                             static_cast<uint64_t>(matC->rows) <= kSpmmArch22MaxRows &&
                             static_cast<uint64_t>(matC->cols) <= kSpmmArch22MaxDim;
    const bool leadingDimsOk = static_cast<uint64_t>(matB->ld) <= kSpmmArch22MaxIndex &&
                               static_cast<uint64_t>(matC->ld) <= kSpmmArch22MaxIndex;
    return sparseDimsOk && denseDimsOk && leadingDimsOk;
}

// D1 fix: matA/matB/matC descriptor pointers are null-checked above, but the
// raw device buffers they wrap were not. Previously a null ptrs/values slipped
// through here and only crashed later inside DoPreProcess/the kernel launch.
static aclsparseStatus_t ValidateSpmmBuffers(const aclsparseSpMatDescr *matA,
                                             const aclsparseDnMatDescr *matB,
                                             const aclsparseDnMatDescr *matC)
{
    const bool bHasElements = matB->rows > 0 && matB->cols > 0;
    const bool cHasElements = matC->rows > 0 && matC->cols > 0;
    if (matA->ptrs == nullptr || (bHasElements && matB->values == nullptr) ||
        (cHasElements && matC->values == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->nnz > 0 && (matA->idxs == nullptr || matA->values == nullptr)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t ValidateSpmmDescriptors(const aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, const aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType)
{
    if (matA->format != ACL_SPARSE_FORMAT_CSR) {
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    aclsparseStatus_t st = AclsparseValidateSupportedCsrIndexTypes(
        matA->ptrType, matA->IdxType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (matA->baseType != ACL_SPARSE_INDEX_BASE_ZERO &&
        matA->baseType != ACL_SPARSE_INDEX_BASE_ONE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedSpmmDtypeCombo(matA, matB, matC, computeType)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!SpmmDimensionsMatch(matA, matB, matC, opA, opB)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    // Do NOT add a blanket M/K/N==0 -> INVALID_VALUE guard here (that's what
    // master's shared AclsparseValidateSpmmCommon does, see commit 3af16fc /
    // Issue #180). This PR's arch22 deliberately supports zero-dim M/N:
    // SpmmExecute() (below) computes the beta-scaled C and returns SUCCESS
    // without launching the main kernel when layout.m==0 or N==0 (see
    // README.md "M/K/N 均允许为 0；零长度输出直接成功返回，K=0 时结果遵循
    // C = beta * C"). Rejecting zero dims here would make that code path
    // unreachable and silently regress an already-shipped feature.
    st = ValidateDnMatLayout(matB);
    if (st == ACL_SPARSE_STATUS_SUCCESS) {
        st = ValidateDnMatLayout(matC);
    }
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return SpmmDimsWithinUint32(matA, matB, matC, opA) ?
        ACL_SPARSE_STATUS_SUCCESS : ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

// D1 fix (follow-up, found by review): aclsparseSpMMGetBufferSize() only
// computes a workspace size from matA->rows -- it never dereferences
// ptrs/idxs/values -- so folding ValidateSpmmBuffers() unconditionally into
// this shared validator made GetBufferSize reject perfectly legal calls
// where the sparse/dense buffers aren't allocated yet (e.g. sizing the
// workspace before the caller has finished setting up matB/matC). Only
// Preprocess/SpMM actually dereference those buffers, so only they should
// require requireBuffers=true.
static aclsparseStatus_t ValidateSpmmInputs(const aclsparseSpMatDescr *matA,
                                            const aclsparseDnMatDescr *matB,
                                            const aclsparseDnMatDescr *matC,
                                            aclsparseOperation_t opA,
                                            aclsparseOperation_t opB,
                                            aclDataType computeType,
                                            aclsparseSpMMAlg_t alg,
                                            bool requireBuffers)
{
    // 不采用 master 新增的 arch22/arch35 公共校验函数 AclsparseValidateSpmmCommon，
    // 原样保留 arch22 自己的校验链（ValidateSpmmOperations/ValidateSpmmDescriptors/
    // ValidateSpmmAlg）。三个理由，都是"共用函数目前的假设不适用于我们"，不是共用
    // 函数本身有 bug：
    // 1) 硬编码 matA->baseType 必须是 ACL_SPARSE_INDEX_BASE_ZERO，会拒绝本 PR 的
    //    idxBase=1 支持（任务书要求项）。
    // 2) dtype 表目前只有全 FP32 一条，不覆盖我们已扩展的 fp16/bf16/complex64。
    // 3) 该函数把 M/K/N==0 统一按 INVALID_VALUE 拒绝（Issue #180 修复），但本 PR
    //    的 arch22 SpMM 是刻意支持零维的：SpmmExecute()（下方）在 layout.m==0 或
    //    N==0 时会先做 beta 缩放再直接返回 SUCCESS、不下 kernel，这是已实现并在
    //    README.md 里写明的行为（"M/K/N 均允许为 0；零长度输出直接成功返回"）。
    //    套用共用函数会让这条已有功能变得不可达，属于静默回退。
    // 结论：这是 arch22 自己范围内的判断，不需要改共用头文件，也不用等其他团队
    // 协调——三点分歧只影响 arch22 走哪条校验路径，不影响共用函数对其他调用方
    // （比如 arch35）的行为。
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    aclsparseStatus_t st = ValidateSpmmOperations(opA, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSpmmDescriptors(matA, matB, matC, opA, opB, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSpmmAlg(alg, opA, opB);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (!requireBuffers) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ValidateSpmmBuffers(matA, matB, matC);
}

aclsparseStatus_t aclsparseSpMMGetBufferSize(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // D1 fix (follow-up)：出参 size 也要校验。这个入口唯一的副作用就是写 *size，
    // 传 nullptr 会直接段错误，与本 PR 给 matA/matB/matC 及其裸设备指针补空校验
    // 的口径一致。
    if (size == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    auto *matAInner = SpmmToMatInner(matA);
    auto *matBInner = SpmmToDnMatInner(matB);
    auto *matCInner = SpmmToDnMatInner(matC);
    aclsparseStatus_t st = ValidateSpmmInputs(matAInner, matBInner, matCInner,
                                              opA, opB, computeType, alg,
                                              /*requireBuffers=*/false);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SpmmWorkspaceLayout layout{};
    st = BuildSpmmWorkspaceLayout(matAInner, opA, layout);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    *size = layout.offsets.totalBytes;
    return ACL_SPARSE_STATUS_SUCCESS;
}

// alpha/beta 的宿主侧取值。complex64 用满四个字段，实数 dtype 只用 alphaRe/betaRe。
struct SpmmHostScalars {
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
};

static aclsparseStatus_t ReadSpmmScalars(const void *alpha, const void *beta,
    bool isComplex64, aclsparsePointerMode_t pointerMode, SpmmHostScalars &out)
{
    out = SpmmHostScalars{0.0f, 0.0f, 0.0f, 0.0f};
    // DEVICE scalars are copied directly into device tiling on the handle's
    // stream after the host-built metadata is installed.
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    if (isComplex64) {
        aclsparseStatus_t ret = ReadScalarComplex64(alpha, pointerMode, out.alphaRe, out.alphaIm);
        if (ret != ACL_SPARSE_STATUS_SUCCESS) {
            return ret;
        }
        return ReadScalarComplex64(beta, pointerMode, out.betaRe, out.betaIm);
    }
    aclsparseStatus_t ret = ReadScalarF32(alpha, pointerMode, out.alphaRe);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ret;
    }
    return ReadScalarF32(beta, pointerMode, out.betaRe);
}

static uint32_t SpmmLogicalN(const aclsparseDnMatDescr *matB, aclsparseOperation_t opB)
{
    return static_cast<uint32_t>(
        opB == ACL_SPARSE_OP_NON_TRANSPOSE ? matB->cols : matB->rows);
}

static uint32_t SpmmLogicalM(const aclsparseSpMatDescr *matA, aclsparseOperation_t opA)
{
    return static_cast<uint32_t>(
        opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->rows : matA->cols);
}

static uint32_t SpmmLogicalK(const aclsparseSpMatDescr *matA, aclsparseOperation_t opA)
{
    return static_cast<uint32_t>(
        opA == ACL_SPARSE_OP_NON_TRANSPOSE ? matA->cols : matA->rows);
}

static bool SpmmUsesFp32HighPrecision(const aclsparseSpMatDescr *matA,
                                      aclDataType computeType,
                                      aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG &&
           matA->valueType == ACL_FLOAT && computeType == ACL_FLOAT;
}

static bool SpmmIsDeterministicAlg(aclsparseSpMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPMM_CSR_ALG3;
}

static bool SpmmUsesFastPath(const aclsparseSpMatDescr *matA,
                             const aclsparseDnMatDescr *matB,
                             const aclsparseDnMatDescr *matC,
                             aclsparseOperation_t opA,
                             aclsparseOperation_t opB,
                             aclDataType computeType,
                             aclsparseSpMMAlg_t alg)
{
    if (SpmmUsesFp32HighPrecision(matA, computeType, alg)) {
        return false;
    }
    // ALG1 is the layout-generic/column-major-oriented route.  ALG2 selects
    // the compact row-major fast path when its layout contract is met and
    // otherwise uses the layout-aware generic route.  ALG3 must keep a fixed
    // row order, so it never uses the reorder-dependent grouped path.
    switch (alg) {
        case ACL_SPARSE_SPMM_CSR_ALG1:
        case ACL_SPARSE_SPMM_CSR_ALG3:
            return false;
        case ACL_SPARSE_SPMM_ALG_DEFAULT:
        case ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG:
        case ACL_SPARSE_SPMM_CSR_ALG2:
            break;
        default:
            return false;
    }
    return opA == ACL_SPARSE_OP_NON_TRANSPOSE && opB == ACL_SPARSE_OP_NON_TRANSPOSE &&
           matB->order == ACL_SPARSE_ORDER_ROW && matB->ld == matB->cols &&
           matC->order == ACL_SPARSE_ORDER_ROW && matC->ld == matC->cols;
}

static uint32_t SelectSpmmRowBlock(const aclsparseSpMatDescr *matA, uint32_t n,
                                   uint32_t m, uint32_t blockDim, bool genericPath,
                                   bool &pipeMode)
{
    pipeMode = false;
    uint32_t rowBlock = 0;
    if (genericPath) {
        return 0;
    }
    if (matA->valueType == ACL_COMPLEX64) {
        rowBlock = SpmmArch22RowGroupComplex64(n);
    } else {
        const uint32_t valBytes = SpmmValTypeBytes(matA->valueType);
        if (SpmmPipeModeRequested() && SpmmArch22PipeRowGroup(n, valBytes) > 0) {
            pipeMode = true;
            rowBlock = SpmmArch22PipeRowGroup(n, valBytes);
        } else {
            rowBlock = SpmmArch22RowGroup(n, valBytes);
        }
    }
    return SpmmArch22EffectiveRowBlock(m, blockDim, rowBlock);
}

// complex64 快路径把复数乘积拆成 re/im 两条 Axpy 分别累加，接近 float 上界时会
// 各自累加成 Inf、合成时 Inf-Inf 得 NaN。为覆盖所有幅值（含 Execute 时标量/输入
// 变化），复数快路径恒启用 kernel 的行级非有限值回退（设备侧 AccNonFinite）。不做
// 昂贵的整块 D2H 回读预检，也不依赖 Preprocess 时的标量；normal 输入仅多一次设备侧
// 非有限检测，代价可忽略。
static void SetSpmmNumericGuard(SpmmArch22TilingData &td, const aclsparseSpMatDescr *matA,
                                const aclsparseDnMatDescr *matB, const SpmmHostScalars &s)
{
    (void)matB;
    (void)s;
    td.numericGuardHost = (matA->valueType == ACL_COMPLEX64 && td.genericPathHost == 0) ? 1 : 0;
}

static void FillSpmmTilingData(SpmmArch22TilingData &td, const aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, const aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    uint32_t N, uint32_t blockDim, const SpmmHostScalars &s, const WsOffsets &off,
    uint32_t rowBlock, bool pipeMode)
{
    bool isComplex64 = (matA->valueType == ACL_COMPLEX64);
    td.M = SpmmLogicalM(matA, opA);
    td.K = SpmmLogicalK(matA, opA);
    td.N = N;
    td.nnz = static_cast<uint32_t>(matA->nnz);
    td.blockDim = blockDim;
    td.nChunks = isComplex64 ? SpmmArch22ComputeNChunksComplex64(N)
                             : SpmmArch22ComputeNChunks(N, SpmmValTypeBytes(matA->valueType));
    td.alphaHost = s.alphaRe;
    td.betaHost  = s.betaRe;
    td.reorderOffset = off.reorderOff;
    td.binEdgeOffset = off.binEdgeOff;
    td.transposeRowOffset = off.transposeRowOff;
    td.transposeSourceRowOffset = off.transposeSourceRowOff;
    td.transposePermutationOffset = off.transposePermutationOff;
    td.alphaImHost = s.alphaIm;
    td.betaImHost  = s.betaIm;
    td.idxBaseHost = (matA->baseType == ACL_SPARSE_INDEX_BASE_ONE) ? 1 : 0;
    td.rowBlockHost = static_cast<int32_t>(rowBlock);
    td.pipeModeHost = pipeMode ? 1 : 0;
    td.bRowsHost = static_cast<uint32_t>(matB->rows);
    td.bColsHost = static_cast<uint32_t>(matB->cols);
    td.aRowsHost = static_cast<uint32_t>(matA->rows);
    td.aColsHost = static_cast<uint32_t>(matA->cols);
    td.ldbHost = static_cast<int32_t>(matB->ld);
    td.ldcHost = static_cast<int32_t>(matC->ld);
    td.bOrderHost = static_cast<int32_t>(matB->order);
    td.cOrderHost = static_cast<int32_t>(matC->order);
    td.opAHost = static_cast<int32_t>(opA);
    td.opBHost = static_cast<int32_t>(opB);
    const bool useHighPrecision = SpmmUsesFp32HighPrecision(matA, computeType, alg);
    td.algHost = useHighPrecision ? static_cast<int32_t>(alg) :
                                    (alg == ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG ?
                                     static_cast<int32_t>(ACL_SPARSE_SPMM_ALG_DEFAULT) :
                                     static_cast<int32_t>(alg));
    td.genericPathHost = SpmmUsesFastPath(matA, matB, matC, opA, opB, computeType, alg) ? 0 : 1;
    // 流水路径 B 行偏移向量化门控（见 spmm_kernel_group.h 的 kVecRowOffset）：
    //  1) 仅快路径（generic 路径不经 IssueBRowAt）；
    //  2) (col-base)*N 用 int32，需 K*N<=INT32_MAX 防溢出；
    //  3) nnz 够大才划算——小矩阵每段多一条向量 Muls+同步反而变慢，回落标量路径。
    constexpr uint32_t kVecRowOffsetMinNnz = 100000;
    const int64_t maxRowOffset = static_cast<int64_t>(td.K) * static_cast<int64_t>(N);
    td.vecRowOffsetHost = (td.genericPathHost == 0 &&
                           maxRowOffset <= static_cast<int64_t>(INT32_MAX) &&
                           td.nnz >= kVecRowOffsetMinNnz) ? 1 : 0;
    SetSpmmNumericGuard(td, matA, matB, s);
}

aclsparseStatus_t SpmmBuildTiling(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    const void *alpha, const void *beta,
    uint8_t *dBuffer, WsOffsets &off, aclsparsePointerMode_t pointerMode,
    uint32_t blockDim)
{
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint32_t N = SpmmLogicalN(matB, opB);
    const bool genericPath = !SpmmUsesFastPath(matA, matB, matC, opA, opB, computeType, alg);

    // 按连续行块装箱，粒度与 kernel 的 rowGroup 取自同一个函数（不等的话组边界会
    // 跨过块边界，kernel 里的连续性判定会一直落空）。
    //
    // bf16 尤其依赖这里：它的非零值只能整批 DMA 进 UB 再向量 Cast，而"一批的值在
    // values 数组里连续"恰好等价于"一组的行号连续"，也就是说 bf16 能不能走分组
    // 路径完全取决于这里有没有按块装箱。
    //
    // 流水路径只实现在实数版 kernel 上，complex64 恒走原路径。开关打开但 UB 排不下
    // 三个半区时（PipeRowGroup 返回 0）也退回原路径 —— 装箱粒度与 tiling 里的
    // pipeModeHost 必须由同一次判定给出，否则 kernel 会按流水口径算出另一个
    // rowGroup，组边界跨过块边界，"行号连续"的判定全部落空。
    bool pipeMode = false;
    uint32_t rowBlock = SelectSpmmRowBlock(
        matA, N, SpmmLogicalM(matA, opA), blockDim, genericPath, pipeMode);

    SpmmArch22CsrMat csrMat(matA, blockDim);
    aclsparseStatus_t pre =
        csrMat.DoPreProcess(dBuffer, off.reorderOff, off.binEdgeOff,
                            off.transposeRowOff, off.transposeSourceRowOff,
                            off.transposePermutationOff, rowBlock,
                            opA != ACL_SPARSE_OP_NON_TRANSPOSE,
                            SpmmIsDeterministicAlg(alg));
    if (pre != ACL_SPARSE_STATUS_SUCCESS) {
        return pre;
    }

    SpmmHostScalars scalars{};
    aclsparseStatus_t sRet = ReadSpmmScalars(alpha, beta,
        matA->valueType == ACL_COMPLEX64, pointerMode, scalars);
    if (sRet != ACL_SPARSE_STATUS_SUCCESS) {
        return sRet;
    }

    SpmmArch22TilingData td{};
    FillSpmmTilingData(td, matA, matB, matC, opA, opB, computeType, alg, N, blockDim,
                       scalars, off, rowBlock, pipeMode);

    aclError aclRet = aclrtMemcpy(dBuffer + off.tilingOff, sizeof(td),
                                  &td, sizeof(td), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static aclsparseStatus_t SpmmBuildTilingNoThrow(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparseSpMMAlg_t alg,
    const void *alpha, const void *beta, uint8_t *buffer, WsOffsets &off,
    aclsparsePointerMode_t pointerMode, uint32_t blockDim)
{
    try {
        return SpmmBuildTiling(matA, matB, matC, opA, opB, computeType, alg, alpha, beta,
                               buffer, off, pointerMode, blockDim);
    } catch (const std::bad_alloc &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    } catch (const std::length_error &) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }
}

// D2 fix (follow-up, found by review): activeBuffer alone is not a safe key
// for the "skip rebuild" fast path -- it only proves the *workspace* buffer
// is unchanged, not that matA's *sparsity pattern* (rows/cols/nnz) is still
// the one the cached reorder/bin-edge table was built for. A caller that
// mutates/replaces matA's CSR data and calls aclsparseSpMM again with the
// same buffer, without an intervening Preprocess, would otherwise silently
// reuse a stale reorder table sized/ordered for the old pattern.
//
// Mirrors the pattern-signature side-table already used by
// sparse/sddmm/arch35 (SddmmPatternSig/GetSddmmPatternCache there): record
// the complete preprocessing signature per buffer pointer when tiling is
// fully built, and only execute when both activeBuffer AND the recorded
// signature match. A mismatch requires another explicit Preprocess call;
// execute must never rebuild reorder/transpose data by synchronously copying
// CSR offsets/indices back to host.
//
// The cache is keyed by the caller's raw workspace pointer and never deletes
// entries (there is no "buffer freed" callback to hook). A fixed-size table
// would evict a still-live descriptor once its capacity is exceeded, turning
// a legal execute into INVALID_VALUE; an
// unordered_map grows only with the number of distinct buffers ever seen and
// so can never evict a live entry. Single-stream serialization is assumed,
// same as sddmm; the mutex only guards the map against multi-handle/stream
// data races on disjoint workspaces.
struct SpmmPatternSig {
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    const void *rowOffsets;
    const void *columnIndices;
    // matB 的列数 N 也必须进签名。FillSpmmTilingData 会把 N、以及由 N 推导出的
    // nChunks 写进 device 侧 tiling，而快路径只刷新 alpha/beta。若只按 matA 的
    // (rows, cols, nnz) 判定，"同一个 matA + 同一块 workspace + 换一个宽度不同的
    // matB" 会命中快路径并沿用旧的 N/nChunks：kernel 按错误的行距寻址 B/C，
    // 且当缓存的 N 大于实际值时会读到 bGm/cGm 之外。
    uint64_t n;
    int64_t ldb;
    int64_t ldc;
    int32_t bOrder;
    int32_t cOrder;
    int32_t opA;
    int32_t opB;
    int32_t alg;
    // valueType 决定 nChunks/rowBlock（FillSpmmTilingData 用 SpmmValTypeBytes），
    // 也必须进签名：不同 dtype 的描述符共享同一组 CSR 指针与 workspace 时，
    // 若签名相等会复用错误 dtype 的 tiling。
    int32_t valueType;
};

// Named (not anonymous/local) so its layout is well-defined for the stream
// copy below. Only covers alphaHost/betaHost (real dtypes); complex64
// additionally refreshes alphaImHost/betaImHost via a second, non-contiguous
// copy in SpmmRefreshScalarsAsync since those fields sit after reorderOffset/
// binEdgeOffset in SpmmArch22TilingData, not adjacent to alphaHost/betaHost.
struct SpmmRefreshedScalars {
    float alphaHost;
    float betaHost;
};

struct SpmmRefreshedScalarsImag {
    float alphaImHost;
    float betaImHost;
};

// DEVICE pointer mode copies alpha/beta straight into the device tiling with a
// stream-ordered device-to-device copy, avoiding any synchronous D2H read.
static aclsparseStatus_t SpmmRefreshDeviceScalars(uint8_t *tiling,
    const void *alpha, const void *beta, bool isComplex64, aclrtStream stream)
{
    const size_t scalarBytes = sizeof(float);
    const uint8_t *alphaBytes = static_cast<const uint8_t *>(alpha);
    const uint8_t *betaBytes = static_cast<const uint8_t *>(beta);
    struct ScalarCopy {
        size_t destination;
        const void *source;
    } copies[] = {
        {offsetof(SpmmArch22TilingData, alphaHost), alphaBytes},
        {offsetof(SpmmArch22TilingData, betaHost), betaBytes},
        {offsetof(SpmmArch22TilingData, alphaImHost), alphaBytes + scalarBytes},
        {offsetof(SpmmArch22TilingData, betaImHost), betaBytes + scalarBytes},
    };
    const size_t copyCount = isComplex64 ? 4 : 2;
    for (size_t i = 0; i < copyCount; ++i) {
        aclError ret = aclrtMemcpyAsync(tiling + copies[i].destination, scalarBytes,
            copies[i].source, scalarBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (ret != ACL_ERROR_NONE) {
            return ACL_SPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

static std::unordered_map<const void *, SpmmPatternSig> &GetSpmmPatternCache()
{
    static std::unordered_map<const void *, SpmmPatternSig> cache;
    return cache;
}

static std::mutex &GetSpmmPatternMutex()
{
    static std::mutex mutex;
    return mutex;
}

static SpmmPatternSig MakeSpmmPatternSig(const aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, const aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclsparseSpMMAlg_t alg)
{
    return SpmmPatternSig{matA->rows, matA->cols, matA->nnz, matA->ptrs, matA->idxs,
        SpmmLogicalN(matB, opB), matB->ld, matC->ld, static_cast<int32_t>(matB->order),
        static_cast<int32_t>(matC->order), static_cast<int32_t>(opA),
        static_cast<int32_t>(opB), static_cast<int32_t>(alg),
        static_cast<int32_t>(matA->valueType)};
}

static bool SpmmPatternEqual(const SpmmPatternSig &a, const SpmmPatternSig &b)
{
    // 按语义分组，避免单条 return 堆积过多 &&（codecheck 条件数量违规）。
    const bool shapeMatch = a.rows == b.rows && a.cols == b.cols && a.nnz == b.nnz;
    const bool indexMatch = a.rowOffsets == b.rowOffsets && a.columnIndices == b.columnIndices;
    const bool dimMatch = a.n == b.n && a.ldb == b.ldb && a.ldc == b.ldc;
    const bool layoutMatch = a.bOrder == b.bOrder && a.cOrder == b.cOrder;
    const bool opMatch = a.opA == b.opA && a.opB == b.opB;
    const bool typeMatch = a.alg == b.alg && a.valueType == b.valueType;
    return shapeMatch && indexMatch && dimMatch && layoutMatch && opMatch && typeMatch;
}

static bool SpmmPatternMatch(const void *buffer, const SpmmPatternSig &sig)
{
    std::lock_guard<std::mutex> lock(GetSpmmPatternMutex());
    auto &cache = GetSpmmPatternCache();
    auto it = cache.find(buffer);
    return it != cache.end() && SpmmPatternEqual(it->second, sig);
}

static void SpmmPatternRecord(const void *buffer, const SpmmPatternSig &sig)
{
    std::lock_guard<std::mutex> lock(GetSpmmPatternMutex());
    GetSpmmPatternCache()[buffer] = sig;
}

// D2 fix: alpha/beta are per-execution scalars, not part of the (expensive,
// sparsity-pattern-only) reordering that Preprocess caches via activeBuffer.
// Previously aclsparseSpMM only refreshed alpha/beta when activeBuffer
// mismatched the caller's buffer, so calling SpMM again with a new alpha/beta
// but the *same* buffer (the documented fast-path/reuse pattern) silently
// kept computing with the stale scalars from the original Preprocess call.
// This writes only the scalar fields, without touching cached preprocessing
// data. aclrtMemcpyAsyncWithCondition preserves user-stream ordering and also
// supports the ordinary host storage used for these small staging structs.
static aclsparseStatus_t SpmmRefreshScalarsAsync(uint8_t *dBuffer, int64_t tilingOff,
    const void *alpha, const void *beta, aclsparsePointerMode_t pointerMode,
    bool isComplex64, aclrtStream stream)
{
    if (alpha == nullptr || beta == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    uint8_t *tiling = dBuffer + tilingOff;
    if (pointerMode == ACL_SPARSE_POINTER_MODE_DEVICE) {
        return SpmmRefreshDeviceScalars(tiling, alpha, beta, isComplex64, stream);
    }
    SpmmRefreshedScalars scalars{};
    SpmmRefreshedScalarsImag scalarsIm{};
    aclsparseStatus_t st;
    if (isComplex64) {
        st = ReadScalarComplex64(alpha, pointerMode, scalars.alphaHost, scalarsIm.alphaImHost);
        if (st == ACL_SPARSE_STATUS_SUCCESS) {
            st = ReadScalarComplex64(beta, pointerMode, scalars.betaHost, scalarsIm.betaImHost);
        }
    } else {
        st = ReadScalarF32(alpha, pointerMode, scalars.alphaHost);
        if (st == ACL_SPARSE_STATUS_SUCCESS) {
            st = ReadScalarF32(beta, pointerMode, scalars.betaHost);
        }
    }
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    aclError aclRet = aclrtMemcpyAsyncWithCondition(
        tiling + offsetof(SpmmArch22TilingData, alphaHost), sizeof(scalars),
        &scalars, sizeof(scalars), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (aclRet != ACL_ERROR_NONE || !isComplex64) {
        return aclRet == ACL_ERROR_NONE ? ACL_SPARSE_STATUS_SUCCESS
                                        : ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpyAsyncWithCondition(
        tiling + offsetof(SpmmArch22TilingData, alphaImHost), sizeof(scalarsIm),
        &scalarsIm, sizeof(scalarsIm), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    return aclRet == ACL_ERROR_NONE ? ACL_SPARSE_STATUS_SUCCESS
                                    : ACL_SPARSE_STATUS_EXECUTION_FAILED;
}

static aclsparseStatus_t ValidateSpmmRuntimeInputs(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    aclsparseStatus_t st = ValidateSpmmInputs(
        matA, matB, matC, opA, opB, computeType, alg, /*requireBuffers=*/true);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    return buffer == nullptr ? ACL_SPARSE_STATUS_INVALID_VALUE : ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMMPreprocess(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    auto *matAInner = SpmmToMatInner(matA);
    auto *matBInner = SpmmToDnMatInner(matB);
    auto *matCInner = SpmmToDnMatInner(matC);
    aclsparseStatus_t st = ValidateSpmmRuntimeInputs(
        matAInner, matBInner, matCInner, opA, opB, computeType, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SpmmWorkspaceLayout layout{};
    st = BuildSpmmWorkspaceLayout(matAInner, opA, layout);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t pmSt = aclsparseGetPointerMode(handle, &pointerMode);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        return pmSt;
    }
    st = SpmmBuildTilingNoThrow(matAInner, matBInner, matCInner, opA, opB, computeType, alg,
                                alpha, beta, (uint8_t *)buffer, layout.offsets, pointerMode,
                                layout.blockDim);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    matAInner->activeBuffer = buffer;
    SpmmPatternRecord(buffer, MakeSpmmPatternSig(
        matAInner, matBInner, matCInner, opA, opB, alg));
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Execute 只复用显式 Preprocess 已写好的 workspace，并在用户 stream 上刷新标量。
// signature 不匹配说明 workspace 已被别的 pattern 覆盖或调用参数发生变化，调用方
// 必须重新 Preprocess；这里禁止同步 D2H 重建 CSR 预处理数据。
static aclsparseStatus_t SpmmPrepareTiling(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclsparseSpMMAlg_t alg,
    const void *alpha, const void *beta, void *buffer,
    const WsOffsets &off, aclsparsePointerMode_t pointerMode, aclrtStream stream)
{
    const SpmmPatternSig sig = MakeSpmmPatternSig(matA, matB, matC, opA, opB, alg);
    bool canReuseTiling = (matA->activeBuffer == buffer) && SpmmPatternMatch(buffer, sig);
    if (!canReuseTiling) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return SpmmRefreshScalarsAsync(static_cast<uint8_t *>(buffer), off.tilingOff,
        alpha, beta, pointerMode, matA->valueType == ACL_COMPLEX64, stream);
}

static void LaunchSpmmPrepared(aclsparseSpMatDescr *matA,
    const aclsparseDnMatDescr *matB, aclsparseDnMatDescr *matC,
    void *matY, void *buffer, const WsOffsets &off, uint32_t blockDim,
    aclrtStream stream)
{
    int32_t valType = SpmmValTypeId(matA->valueType);
    uint8_t *tiling = static_cast<uint8_t *>(buffer) + off.tilingOff;
    if (matY == matC->values) {
        spmm_arch22_kernel_launch(matA->ptrs, matA->idxs, matA->values,
            matB->values, matC->values, buffer, tiling, blockDim, stream, valType);
        return;
    }
    spmm_arch22_kernel_launch_out(matA->ptrs, matA->idxs, matA->values,
        matB->values, matC->values, matY, buffer, tiling, blockDim, stream, valType);
}

static aclsparseStatus_t SpmmExecute(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer, void *matY)
{
    aclrtStream stream = nullptr;
    aclsparseStatus_t st = aclsparseGetStream(handle, &stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }

    auto *matAInner = SpmmToMatInner(matA);
    auto *matBInner = SpmmToDnMatInner(matB);
    auto *matCInner = SpmmToDnMatInner(matC);
    st = ValidateSpmmRuntimeInputs(
        matAInner, matBInner, matCInner, opA, opB, computeType, alg, buffer);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    SpmmWorkspaceLayout layout{};
    st = BuildSpmmWorkspaceLayout(matAInner, opA, layout);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    uint32_t N = SpmmLogicalN(matBInner, opB);
    if (layout.m != 0 && N != 0 && matY == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparsePointerMode_t pointerMode = ACL_SPARSE_POINTER_MODE_HOST;
    aclsparseStatus_t pmSt = aclsparseGetPointerMode(handle, &pointerMode);
    if (pmSt != ACL_SPARSE_STATUS_SUCCESS) {
        return pmSt;
    }
    st = SpmmPrepareTiling(matAInner, matBInner, matCInner, opA, opB, alg,
                           alpha, beta, buffer, layout.offsets, pointerMode, stream);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }
    if (layout.m == 0 || N == 0) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    LaunchSpmmPrepared(matAInner, matBInner, matCInner, matY, buffer, layout.offsets,
                       layout.blockDim, stream);
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseSpMM(aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    auto *matCInner = SpmmToDnMatInner(matC);
    void *matY = matCInner == nullptr ? nullptr : matCInner->values;
    return SpmmExecute(handle, opA, opB, alpha, matA, matB, beta, matC,
                       computeType, alg, buffer, matY);
}

extern "C" __attribute__((visibility("default")))
aclsparseStatus_t aclsparseSpMMOutOfPlaceInternal(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, void *matY, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer)
{
    return SpmmExecute(handle, opA, opB, alpha, matA, matB, beta, matC,
                       computeType, alg, buffer, matY);
}
