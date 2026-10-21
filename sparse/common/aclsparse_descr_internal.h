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
 * \file aclsparse_descr_internal.h
 * \brief ops-sparse 描述符内部结构体（稀疏矩阵 / 稠密向量 / 稠密矩阵）。
 *
 * 这些结构体仅在库实现内部可见，对外只前向声明为不透明的 struct，
 * 对外句柄类型 aclsparseSpMatDescr_t / aclsparseDnVecDescr_t / aclsparseDnMatDescr_t
 * 即为指向这些 struct 的指针（含 const 变体）。
 * 与 aclsparseContext 保持一致：用默认成员初始化 + new/delete 管理生命周期，
 * 避免把内部布局暴露到对外头文件、也避免 malloc 后字段未初始化的隐患。
 */

#ifndef ACLSPARSE_DESCR_INTERNAL_H
#define ACLSPARSE_DESCR_INTERNAL_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <acl/acl.h>
#include "cann_ops_sparse.h"

// Installed CANN headers may only expose ACL_SPARSE_SPGEMM_ALG_DEFAULT as an
// enum enumerator (not a macro). #ifndef is required because #defined(enum)
// is always false.
#ifndef ACL_SPARSE_SPGEMM_DEFAULT
#define ACL_SPARSE_SPGEMM_DEFAULT ACL_SPARSE_SPGEMM_ALG_DEFAULT
#endif

// 描述符签名常量，用于在 Destroy/Get/Set/Execute 前识别不透明句柄的描述符类型，
// 并拒绝错误类型的句柄。
enum DescrSignature : uint32_t {
    kSpMatSignature = 0x53504D54,
    kDnVecSignature = 0xD0D2D4D6,
    kSpVecSignature = 0x51705663,
    kDnMatSignature = 0xD0D2D4D8,
    kSpGemmSignature = 0x5350474D,  // ASCII "SPGM"，表示 SpGEMM 描述符类型。
};

enum class AclsparseSpGemmState : uint32_t {
    CREATED = 0,
    WORK_SIZE_QUERIED,
    WORK_ESTIMATED,
    MEMORY_ESTIMATED,
    COMPUTE_SIZE_QUERIED,
    COMPUTED,
    COPIED
};

// Multi-stage SpGEMM descriptor. Device buffers remain caller-owned; the
// descriptor only keeps weak references needed to enforce stage order and to
// let Copy consume the exact intermediate data produced by Compute.
struct aclsparseSpGEMMDescr {
    uint32_t signature = 0;
    AclsparseSpGemmState state = AclsparseSpGemmState::CREATED;
    aclsparseSpGEMMAlg_t alg = ACL_SPARSE_SPGEMM_DEFAULT;
    aclDataType computeType = ACL_FLOAT;
    const aclsparseSpMatDescr *matA = nullptr;
    const aclsparseSpMatDescr *matB = nullptr;
    aclsparseSpMatDescr *matC = nullptr;
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t nnzA = 0;
    uint64_t nnzB = 0;
    int64_t numProducts = 0;
    int64_t nnzC = 0;
    int32_t regularDegree = 0;
    bool copyRequired = true;
    bool forceGenericPath = false;
    int32_t cInDataValid = 0;
    size_t requiredBuffer1 = 0;
    size_t requiredBuffer2 = 0;
    void *externalBuffer1 = nullptr;
    void *externalBuffer2 = nullptr;
    float chunkFraction = 1.0F;
};

// 稀疏矩阵描述符内部结构（CSR / CSC / COO / BELL / SLICED_ELL 共用）。
struct aclsparseSpMatDescr {
    uint32_t signature = kSpMatSignature;
    // Non-SpMV preprocess cache (SpMM/SDDMM workspace layout).
    const void *activeBuffer = nullptr;
    // SpMV uses a separate layout and must never populate activeBuffer.
    const void *activeSpmvBuffer = nullptr;
    // Parameters captured by the most recent SpMV preprocess. Keeping
    // these fields beside activeSpmvBuffer prevents a workspace from being reused
    // after the CSR pattern or static execution attributes have changed.
    const void *activeRowOffsets = nullptr;
    const void *activeColIndices = nullptr;
    const void *activeValues = nullptr;
    int64_t activeRowOffsetsStride = 1;
    int64_t activeColIndStride = 1;
    int64_t activeValuesStride = 1;
    int64_t activeXStride = 1;
    int64_t activeYStride = 1;
    uint64_t activeRows = 0;
    uint64_t activeCols = 0;
    uint64_t activeNnz = 0;
    aclsparseIndexBase_t activeBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclDataType activeValueType = ACL_FLOAT;
    aclDataType activeXType = ACL_FLOAT;
    aclDataType activeYType = ACL_FLOAT;
    aclDataType activeComputeType = ACL_FLOAT;
    aclsparseOperation_t activeOp = ACL_SPARSE_OP_NON_TRANSPOSE;
    int32_t activeAlg = -1;
    // Preprocess metadata is produced asynchronously on the handle stream.
    // It must not be reused after a stream switch without re-preprocessing.
    const void *activeStream = nullptr;
    uint64_t activeStreamEpoch = 0;
    // DenseToSparse discovers nnz/offsets before callers allocate its output
    // payload. Rebinding indices/values must not discard that analysis, while
    // generic SpMM/SDDMM and SpMV preprocess caches still require invalidation.
    const void *activeDenseToSparseBuffer = nullptr;
    bool denseToSparseAnalysisReady = false;
    int32_t cInDataValid = 0;
    // 旧三阶段 SpGEMM 接口复用新多阶段实现时保存的内部描述符及 workspace。
    // 生命周期随 matC 结束，避免旧接口额外暴露描述符管理要求。
    std::unique_ptr<aclsparseSpGEMMDescr> legacySpGemmDescr;
    const void *legacySpGemmBuffer = nullptr;
    aclsparseFormat_t format{};
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t nnz = 0;
    void *ptrs = nullptr;
    void *idxs = nullptr;
    void *values = nullptr;
    int64_t ptrStride = 1;
    int64_t idxStride = 1;
    int64_t valueStride = 1;
    aclsparseIndexBase_t baseType{};
    aclsparseIndexType_t ptrType{};
    aclsparseIndexType_t IdxType{};
    aclDataType valueType{};
    aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
    aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    uint64_t sliceNnz = 0;
    int64_t numSlices = 0;
    void *rowInds = nullptr;
    void *colInds = nullptr;
    uint64_t ellBlockSize = 0;
    uint64_t ellCols = 0;
    void *ellColInd = nullptr;
};

inline aclsparseStatus_t ValidateSpMatCreateParams(
    const void *descrPtr, int64_t rows, int64_t cols, int64_t nnz)
{
    if (descrPtr == nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (rows < 0 || cols < 0 || nnz < 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 校验 CSR 稀疏矩阵索引类型（严格模式）：当前仅支持 32 位且 ptr/col 类型一致。
// 供 spmm、spmv/arch22 等仅支持 I32 索引的算子在 execute 路径使用。
inline aclsparseStatus_t AclsparseValidateSupportedCsrIndexTypes(aclsparseIndexType_t ptrType,
                                                                 aclsparseIndexType_t idxType)
{
    if (ptrType != idxType) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (ptrType == ACL_SPARSE_INDEX_32I) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}

// 校验 CSR 稀疏矩阵索引类型（扩展模式）：
// ptrType / idxType 可分别使用 I32 或 I64，允许四种组合，拒绝其余类型。
// 供 aclsparseCreateCsr / aclsparseCreateConstCsr 使用。
inline aclsparseStatus_t AclsparseValidateSupportedCsrIndexTypesExtended(aclsparseIndexType_t ptrType,
                                                                         aclsparseIndexType_t idxType)
{
    // ptrType 仅允许 I32 / I64
    if (ptrType != ACL_SPARSE_INDEX_32I && ptrType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // idxType 仅允许 I32 / I64
    if (idxType != ACL_SPARSE_INDEX_32I && idxType != ACL_SPARSE_INDEX_64I) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// 稠密向量描述符内部结构（SpMV 的 x / y）。
struct aclsparseDnVecDescr {
    uint32_t signature = 0;
    uint64_t nums = 0;
    void *values = nullptr;
    int64_t stride = 1;
    aclDataType valueType{};
};

// 稀疏向量描述符内部结构。
struct aclsparseSpVecDescr {
    uint32_t signature = 0;
    uint64_t size = 0;
    uint64_t nnz = 0;
    void *indices = nullptr;
    void *values = nullptr;
    aclsparseIndexType_t idxType{};
    aclsparseIndexBase_t idxBase{};
    aclDataType valueType{};
};

// 稠密矩阵描述符内部结构（SpMM 的 B / C）。
struct aclsparseDnMatDescr {
    uint32_t signature = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    aclsparseOrder_t order{};
    void *values = nullptr;
    aclDataType valueType{};
};

// SpMM 支持的 dtype 组合四元组（matA / matB / matC / computeType）。
// arch22 仅支持全 FP32；arch35 支持统一 FP32 / FP16 / INT8 量化共三组合，
// 各架构以表驱动方式传入公共校验函数。
struct AclsparseSpmmDtypeCombo {
    aclDataType matA;
    aclDataType matB;
    aclDataType matC;
    aclDataType computeType;
};

// SpMM dtype 组合匹配（表驱动）：matA/matB/matC/computeType 四元组与 combos
// 表中任一组合全等即命中。供 AclsparseValidateSpmmCommon 调用，独立成函数以
// 控制单函数圈复杂度。
inline bool AclsparseMatchSpmmDtypeCombo(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclDataType computeType,
    const AclsparseSpmmDtypeCombo *combos, size_t comboCount)
{
    for (size_t i = 0; i < comboCount; ++i) {
        if (matA->valueType == combos[i].matA && matB->valueType == combos[i].matB &&
            matC->valueType == combos[i].matC && computeType == combos[i].computeType) {
            return true;
        }
    }
    return false;
}

// SpMM 入参公共校验（arch22/arch35 共用，消除跨架构重复代码）：
// 非空 + opA/opB 操作类型 + CSR 格式 + 索引类型（严格 I32）+ base-zero +
// dtype 组合（表驱动）+ 维度匹配 + 零维守卫。
// allowOpBTranspose：arch35 支持 opB 转置（N/T），arch22 仅支持 N 时传 false。
// 各架构差异化检查（alg 集合、紧凑布局、INT32_MAX 上限等）由调用方在本函数
// 返回 SUCCESS 后自行补充。
inline aclsparseStatus_t AclsparseValidateSpmmCommon(
    const aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB,
    const aclsparseDnMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclDataType computeType, bool allowOpBTranspose,
    const AclsparseSpmmDtypeCombo *combos, size_t comboCount)
{
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    // SpMM 仅支持 opA = NON_TRANSPOSE
    if (opA != ACL_SPARSE_OP_NON_TRANSPOSE) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != ACL_SPARSE_OP_NON_TRANSPOSE &&
        (!allowOpBTranspose || opB != ACL_SPARSE_OP_TRANSPOSE)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->format != ACL_SPARSE_FORMAT_CSR) {
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    aclsparseStatus_t idxSt = AclsparseValidateSupportedCsrIndexTypes(matA->ptrType, matA->IdxType);
    if (idxSt != ACL_SPARSE_STATUS_SUCCESS) {
        return idxSt;
    }
    if (matA->baseType != ACL_SPARSE_INDEX_BASE_ZERO) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // dtype 组合校验（表驱动，匹配逻辑见 AclsparseMatchSpmmDtypeCombo）
    if (!AclsparseMatchSpmmDtypeCombo(matA, matB, matC, computeType, combos, comboCount)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }
    // 维度匹配 + 零维守卫：维度匹配对 0==0 放行，随后由零维守卫拒绝 m/k/n
    // 任一为 0 的输入（0 维矩阵支持性由算子层声明，spmm 不支持）。
    if (matA->cols != static_cast<uint64_t>(matB->rows)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows != static_cast<uint64_t>(matC->rows) || matB->cols != matC->cols) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->rows == 0 || matA->cols == 0 || matC->cols == 0) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

// Cube SpMM 稀疏矩阵描述符内部结构（Cube-BCSR 专用）。
// 不保存 COO 原始输入；COO 数组在 aclsparseCubeSpmmPreprocess 调用时直接传入。
struct aclsparseCubeSpmmMatDescr {
    // 记录当前与该 Cube-BCSR 描述符绑定的 workspace buffer，用于多次 SpMM 复用。
    const void *activeBuffer = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    int64_t blockM = 16;
    int64_t blockK = 16;
    int32_t numCores = 0;
    void *rwPtr = nullptr;     // device, int64, 预处理输出
    void *colRef = nullptr;    // device, int32, 预处理输出
    void *vals = nullptr;      // device, float16, 预处理输出
    void *coreInfo = nullptr;  // device, int32, 预处理输出
    aclsparseIndexBase_t baseType{};
    aclsparseIndexType_t ptrType{};
    aclsparseIndexType_t IdxType{};
    aclDataType valueType{};
};

// Legacy 矩阵描述符内部结构（Format Conversion / nnz 等 Legacy API 使用）。
struct aclsparseMatDescr {
    aclsparseMatrixType_t type = ACL_SPARSE_MATRIX_TYPE_GENERAL;
    aclsparseIndexBase_t indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    aclsparseDiagType_t diagType = ACL_SPARSE_DIAG_TYPE_NON_UNIT;
    aclsparseFillMode_t fillMode = ACL_SPARSE_FILL_MODE_LOWER;
};

// SpMVOp 描述符内部结构（Generic API aclsparseSpMVOp 使用）。
// 持有 matA 的 weak reference（指针不持有所有权，由用户管理生命周期）。
// ALG1 允许仅更新 csrValues 而不重建 descr/plan。
struct aclsparseSpMVOpDescr {
    aclsparseFormat_t format{ACL_SPARSE_FORMAT_CSR};
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t nnz = 0;
    aclsparseIndexBase_t indexBase{ACL_SPARSE_INDEX_BASE_ZERO};
    aclsparseIndexType_t rowOffsetType{ACL_SPARSE_INDEX_32I};
    aclsparseIndexType_t colIndType{ACL_SPARSE_INDEX_32I};
    aclDataType valueType{ACL_FLOAT};
    // CSR device pointers（不持有所有权）
    void *csrRowOffsets = nullptr;
    void *csrColInd = nullptr;
    void *csrValues = nullptr;
    // 算法
    aclsparseSpMVOpAlg_t alg{ACL_SPARSE_SPMVOP_ALG_DEFAULT};
    // ALG2 预处理关联的 user-provided buffer（用于 reorder/bin_edge）
    void *userBuffer = nullptr;
    // ALG2: preprocess kernel 使用的 block 数（execute 时复用，避免重新计算导致不一致）
    // ALG1/DEFAULT: 未使用（execute 时独立计算 useBlocks）
    uint32_t alg2NumBlocks = 0;
};

// SpMMOp 描述符内部结构（Generic API aclsparseSpMMOp 使用）。
// 持有 matA 的 weak reference（指针不持有所有权，由用户管理生命周期）。
// matA 在 createDescr 阶段绑定（CSR pattern 固定用于 plan 生命周期）；
// matB/matC 在 execute 阶段传入（可跨多次 execute 更换，n/ldb/ldc/order 可变）。
// ALG1 允许仅更新 csrValues 而不重建 descr/plan。
struct aclsparseSpMMOpDescr {
    aclsparseFormat_t format{ACL_SPARSE_FORMAT_CSR};
    uint64_t m = 0;              // CSR 行数（固定，createDescr 绑定）
    uint64_t k = 0;              // CSR 列数 = 收缩维度（固定，用于 execute 维度校验）
    uint64_t nnz = 0;
    aclsparseIndexBase_t indexBase{ACL_SPARSE_INDEX_BASE_ZERO};  // ZERO 或 ONE
    aclsparseIndexType_t rowOffsetType{ACL_SPARSE_INDEX_32I};     // I32 或 I64（ptrType）
    aclDataType valueType{ACL_FLOAT};  // FP32 或 FP16（A/B/C 一致）
    aclsparseOperation_t opB{ACL_SPARSE_OP_NON_TRANSPOSE};  // createDescr 绑定，plan 生命周期固定
    // CSR device pointers（weak reference，不持有所有权）
    void *csrRowOffsets = nullptr;  // int32_t* 或 int64_t*（由 rowOffsetType 决定）
    void *csrColInd = nullptr;      // int32_t*（IdxType 固定 I32）
    void *csrValues = nullptr;      // float* 或 __fp16*
    // 算法
    aclsparseSpMMOpAlg_t alg{ACL_SPARSE_SPMMOP_ALG_DEFAULT};
    // ALG2 预处理关联的 user-provided buffer（用于 reorder/bin_edge/scratch/tmpScratch）
    void *userBuffer = nullptr;
    // ALG2: preprocess kernel 使用的 block 数（execute 时复用，避免重新计算不一致）
    // ALG1/DEFAULT: 未使用（execute 时独立计算 useBlocks）
    uint32_t alg2NumBlocks = 0;
};

#endif // ACLSPARSE_DESCR_INTERNAL_H
