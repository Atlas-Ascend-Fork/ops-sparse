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
 * \file aclsparselt_matmul_descriptor.cpp
 * \brief ops-sparseLt matmul 描述符初始化与销毁函数实现。
 */

#include "cann_ops_sparseLt.h"
#include "aclsparselt_matmul_descriptor_internal.h"
#include "shared/aclsparselt_internal.h" // kSparseLtLogTag + log/log.h（OP_LOGE）

#include <new>

namespace {

/// C/D 矩阵 rows/cols 维度上限
constexpr int64_t kMaxDim = 2097120;

/**
 * @brief 判断 op 是否为合法的 aclsparseOperation_t 枚举值。
 *
 * descriptor init 阶段仅支持 NON_TRANSPOSE / TRANSPOSE，
 * 不支持 CONJUGATE_TRANSPOSE。
 */
inline bool IsValidOperation(aclsparseOperation_t op)
{
    return op == ACL_SPARSE_OP_NON_TRANSPOSE ||
           op == ACL_SPARSE_OP_TRANSPOSE;
}

/**
 * @brief 判断 computeType 是否为合法的 aclsparseComputeType_t 枚举值。
 */
inline bool IsValidComputeType(aclsparseComputeType_t t)
{
    return t == ACL_SPARSE_COMPUTE_16F ||
           t == ACL_SPARSE_COMPUTE_32F ||
           t == ACL_SPARSE_COMPUTE_32I;
}

/**
 * @brief 校验 computeType 与 valueType 的绑定关系（对齐 cuSPARSELt v0.10.0
 *        数据类型表）。
 *
 * 绑定表：32F ↔ FP32/FP16/BF16；16F ↔ 仅 FP16（超集：按 FP32 累加执行，
 * 精度与 32F 路径一致、优于 cuSPARSELt 16F 的半精度累加，见 matmul
 * README）；32I ↔ INT8。主四型之外的类型（FP8/FP4/HIFLOAT8/INT32 等）跳过
 * 校验——描述符层保持 A8 超集行为，执行层 dtype 白名单
 * （matmul_host.cpp validate_launch_dtypes）照旧拒绝。
 */
inline bool IsComputeTypeMatchedWithDtype(aclsparseComputeType_t ct, aclDataType vt)
{
    switch (vt) {
        case ACL_FLOAT:    return ct == ACL_SPARSE_COMPUTE_32F;
        case ACL_FLOAT16:  return ct == ACL_SPARSE_COMPUTE_32F || ct == ACL_SPARSE_COMPUTE_16F;
        case ACL_BF16:     return ct == ACL_SPARSE_COMPUTE_32F;
        case ACL_INT8:     return ct == ACL_SPARSE_COMPUTE_32I;
        default:           return true;   // A8 超集域：描述符层跳过，执行层照旧拒绝
    }
}

/**
 * @brief 判断 valueType 是否属于 INT8/FP8/FP4 低精度档（需施加操作布局组合约束）。
 *
 * 1 字节高精度类型 HIFLOAT8 不在此档，与 FP32/FP16/BF16 一样不受组合约束。
 */
inline bool IsLowPrecisionType(aclDataType t)
{
    return t == ACL_INT8 ||
           t == ACL_FLOAT8_E4M3FN ||
           t == ACL_FLOAT8_E5M2 ||
           t == ACL_FLOAT8_E8M0 ||
           t == ACL_FLOAT4_E2M1 ||
           t == ACL_FLOAT4_E1M2;
}

/**
 * @brief 判断矩阵描述符引用是否有效（指针非空且句柄已初始化）。
 */
inline bool IsMatDescrValid(const aclsparseLtMatDescriptor_t* mat)
{
    return mat != nullptr && *mat != nullptr;
}

/**
 * @brief 判断矩阵描述符的 rows/cols 是否在维度上限内（<= 2097120）。
 */
inline bool IsDimWithinLimit(const aclsparseLtMatDescriptor* desc)
{
    return desc->rows <= kMaxDim && desc->cols <= kMaxDim;
}

/**
 * @brief 取 op 的反转（N <-> T）。
 *
 * cuSPARSELt "the opposite" 的语义为逐 op 取反（TN->NT、NN->TT），而非
 * 交换 opA/opB——二者仅在基准对的两个 op 互异时等价，对 (N,N)/(T,T)
 * 基准对交换恒等于自身（方案 §7.2 的 pair-swap 草据此误拒
 * (Row,Col)+TT/(Col,Row)+NN 两个官方合法组合，实测 4 用例在描述符层
 * 即被拒，本函数为设计意图的修正实现）。
 */
inline aclsparseOperation_t OpOpposite(aclsparseOperation_t op)
{
    return (op == ACL_SPARSE_OP_NON_TRANSPOSE) ? ACL_SPARSE_OP_TRANSPOSE
                                               : ACL_SPARSE_OP_NON_TRANSPOSE;
}

/**
 * @brief 判断 (opA, opB) 是否命中基准 op 对或其反转对。
 *
 * 并集口径（超集政策）：基准对 = A 表（A-structured 基准表）的 op 对；
 * 反转对 = 基准对逐 op 取反后的 op 对，即 cuSPARSELt "the opposite
 * if B is structured" 的 B-sparse 合法集（官方原文：opA/opB = TN if
 * orderA/orderB = Col/Col; NT if Row/Row; NN if Row/Col; TT if Col/Row，
 * B 为 structured 时取 opposite）。接受 cuSPARSELt 拒绝的组合属超集政策
 * 允许域（前提结果正确，由 INT8 精度用例实证）；不区分稀疏侧——A-sparse
 * 的反转组合同样放行（官方 A-structured 仅基准表，放行属超集接受）。
 */
inline bool IsOpLayoutPairUnion(aclsparseOperation_t wantA, aclsparseOperation_t wantB,
                                aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    return (opA == wantA && opB == wantB) ||
           (opA == OpOpposite(wantA) && opB == OpOpposite(wantB));
}

/**
 * @brief 校验 INT8/FP8/FP4 类型下的操作与布局组合约束（并集口径）。
 *
 * 缩写：N = ACL_SPARSE_OP_NON_TRANSPOSE, T = ACL_SPARSE_OP_TRANSPOSE。
 * 每个 order 对允许两个 op 对（基准对 + 其反转对，后者为 B-sparse 官方
 * 合法集 S_c；[S_c] 标注）：
 * | orderA | orderB | 基准 (opA,opB) | + 反转对         |
 * | COL    | COL    | (T, N)         | (N, T)  [S_c]   |
 * | ROW    | ROW    | (N, T)         | (T, N)  [S_c]   |
 * | ROW    | COL    | (N, N)         | (T, T)  [S_c]   |
 * | COL    | ROW    | (T, T)         | (N, N)  [S_c]   |
 * 触发键维持 descA->valueType（与官方原文以 matA 类型起算的口径一致；
 * A/B dtype 混搭域已由 computeType 双侧绑定校验封死）。CONJUGATE_TRANSPOSE
 * 已在 IsValidOperation 中被拒绝，不会进入此函数。
 */
inline bool IsValidOpLayoutCombination(aclsparseOrder_t orderA, aclsparseOrder_t orderB,
                                       aclsparseOperation_t opA, aclsparseOperation_t opB)
{
    if (orderA == ACL_SPARSE_ORDER_COL && orderB == ACL_SPARSE_ORDER_COL) {
        return IsOpLayoutPairUnion(ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, opA, opB);
    }
    if (orderA == ACL_SPARSE_ORDER_ROW && orderB == ACL_SPARSE_ORDER_ROW) {
        return IsOpLayoutPairUnion(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, opA, opB);
    }
    if (orderA == ACL_SPARSE_ORDER_ROW && orderB == ACL_SPARSE_ORDER_COL) {
        return IsOpLayoutPairUnion(ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, opA, opB);
    }
    if (orderA == ACL_SPARSE_ORDER_COL && orderB == ACL_SPARSE_ORDER_ROW) {
        return IsOpLayoutPairUnion(ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, opA, opB);
    }
    return false;
}

/**
 * @brief 校验结构化稀疏位置约束。
 *
 * matA 与 matB 有且仅有一个为 structured；matC/matD 须为 dense。
 */
inline aclsparseStatus_t ValidateSparsePosition(
    const aclsparseLtMatDescriptor* descA,
    const aclsparseLtMatDescriptor* descB,
    const aclsparseLtMatDescriptor* descC,
    const aclsparseLtMatDescriptor* descD)
{
    if (descA->isStructured == descB->isStructured) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (descC->isStructured || descD->isStructured) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 校验 C/D 一致性约束：相同 ld 和 order。
 */
inline aclsparseStatus_t ValidateCDConsistency(
    const aclsparseLtMatDescriptor* descC,
    const aclsparseLtMatDescriptor* descD)
{
    if (descC->ld != descD->ld || descC->order != descD->order) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 校验 A/B/C/D 四个矩阵的 numBatches 一致性。
 */
inline aclsparseStatus_t ValidateBatchConsistency(
    const aclsparseLtMatDescriptor* descA,
    const aclsparseLtMatDescriptor* descB,
    const aclsparseLtMatDescriptor* descC,
    const aclsparseLtMatDescriptor* descD)
{
    const int32_t nbA = (descA->numBatches > 0) ? descA->numBatches : 1;
    const int32_t nbB = (descB->numBatches > 0) ? descB->numBatches : 1;
    const int32_t nbC = (descC->numBatches > 0) ? descC->numBatches : 1;
    const int32_t nbD = (descD->numBatches > 0) ? descD->numBatches : 1;
    if (nbA != nbB || nbA != nbC || nbA != nbD) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 校验四矩阵形状链（int64 域直比，不经 md_m/md_k/md_n 的 int32 收窄）。
 *
 * 乘法定义约束（m/n 派生与 tiling 的 md_m/md_n 同口径）：
 *   1) k 链：op(A) 列数 == op(B) 行数；
 *   2) C/D 形状：rows == m 且 cols == n（cuSPARSELt "D has the same shape
 *      of C" + C 形状由乘法定义约束；C/D 的 ld/order 一致已由前置校验保证）。
 * 其中 m = opA==T ? A->cols : A->rows，n = opB==T ? B->rows : B->cols。
 * 缺失该校验时：C/D 小于 (m,n) 会使 epilogue 按 (m,n,ldC) 推导的访问 span
 * 越界读写用户缓冲；B 实际行数 != k 时静默错果。
 */
inline aclsparseStatus_t ValidateShapeChain(
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const aclsparseLtMatDescriptor* descA,
    const aclsparseLtMatDescriptor* descB,
    const aclsparseLtMatDescriptor* descC,
    const aclsparseLtMatDescriptor* descD)
{
    const int64_t kFromA = (opA == ACL_SPARSE_OP_TRANSPOSE) ? descA->rows : descA->cols;
    const int64_t kFromB = (opB == ACL_SPARSE_OP_TRANSPOSE) ? descB->cols : descB->rows;
    const int64_t m = (opA == ACL_SPARSE_OP_TRANSPOSE) ? descA->cols : descA->rows;
    const int64_t n = (opB == ACL_SPARSE_OP_TRANSPOSE) ? descB->rows : descB->cols;
    if (kFromA != kFromB || descC->rows != m || descC->cols != n ||
        descD->rows != m || descD->cols != n) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulDescriptorInit: shape chain mismatch: opA=%d, opB=%d, "
                "A(%lld,%lld), B(%lld,%lld), C(%lld,%lld), D(%lld,%lld); "
                "expect op(A) cols == op(B) rows (k: %lld vs %lld) and C/D == (m,n)=(%lld,%lld)",
                opA, opB,
                static_cast<long long>(descA->rows), static_cast<long long>(descA->cols),
                static_cast<long long>(descB->rows), static_cast<long long>(descB->cols),
                static_cast<long long>(descC->rows), static_cast<long long>(descC->cols),
                static_cast<long long>(descD->rows), static_cast<long long>(descD->cols),
                static_cast<long long>(kFromA), static_cast<long long>(kFromB),
                static_cast<long long>(m), static_cast<long long>(n));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 校验 matmul 描述符的参数与约束。
 *
 * 校验顺序：mat nullptr → opA/opB 枚举 → computeType 枚举 →
 *           computeType↔A/B valueType 匹配 → 结构化稀疏位置 →
 *           C/D 一致性 → batch 一致性 → 形状链 → 操作布局组合 → 维度上下限。
 * 所有校验在内存分配之前完成。
 */
aclsparseStatus_t ValidateMatmulDescriptorParams(
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const aclsparseLtMatDescriptor_t* matA,
    const aclsparseLtMatDescriptor_t* matB,
    const aclsparseLtMatDescriptor_t* matC,
    const aclsparseLtMatDescriptor_t* matD,
    aclsparseComputeType_t computeType)
{
    if (!IsMatDescrValid(matA) || !IsMatDescrValid(matB) ||
        !IsMatDescrValid(matC) || !IsMatDescrValid(matD)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidOperation(opA) || !IsValidOperation(opB)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    if (!IsValidComputeType(computeType)) {
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    const aclsparseLtMatDescriptor* descA = ToInternalConst(matA);
    const aclsparseLtMatDescriptor* descB = ToInternalConst(matB);
    const aclsparseLtMatDescriptor* descC = ToInternalConst(matC);
    const aclsparseLtMatDescriptor* descD = ToInternalConst(matD);

    // computeType↔A/B valueType 绑定（双侧各查一次，对齐官方表——computeType
    // 与 A/B 输入联合绑定；同时封死主四型 A/B dtype 混用的静默错值域：
    // 如 A=FP16+B=INT8 时 computeType 无法同时匹配两侧）。
    if (!IsComputeTypeMatchedWithDtype(computeType, descA->valueType) ||
        !IsComputeTypeMatchedWithDtype(computeType, descB->valueType)) {
        OP_LOGE(kSparseLtLogTag,
                "MatmulDescriptorInit: computeType=%d mismatches A/B dtype "
                "(A=%d, B=%d); binding: 32F<->FP32/FP16/BF16, 16F<->FP16 only, 32I<->INT8",
                static_cast<int32_t>(computeType),
                static_cast<int32_t>(descA->valueType),
                static_cast<int32_t>(descB->valueType));
        return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    }

    aclsparseStatus_t st = ValidateSparsePosition(descA, descB, descC, descD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateCDConsistency(descC, descD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateBatchConsistency(descA, descB, descC, descD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    st = ValidateShapeChain(opA, opB, descA, descB, descC, descD);
    if (st != ACL_SPARSE_STATUS_SUCCESS) { return st; }

    // 操作与布局组合约束（仅 INT8/FP8/FP4 类型，以 matA.valueType 为准）
    if (IsLowPrecisionType(descA->valueType)) {
        if (!IsValidOpLayoutCombination(descA->order, descB->order, opA, opB)) {
            return ACL_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    // 维度上下限约束：matC/matD 的 rows、cols 均 <= 2097120
    if (!IsDimWithinLimit(descC) || !IsDimWithinLimit(descD)) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

/**
 * @brief 填充 matmul 描述符字段。
 */
inline void FillMatmulDescriptor(aclsparseLtMatmulDescriptor* desc,
                                 aclsparseOperation_t opA, aclsparseOperation_t opB,
                                 const aclsparseLtMatDescriptor_t* matA,
                                 const aclsparseLtMatDescriptor_t* matB,
                                 const aclsparseLtMatDescriptor_t* matC,
                                 const aclsparseLtMatDescriptor_t* matD,
                                 aclsparseComputeType_t computeType)
{
    desc->opA = opA;
    desc->opB = opB;
    desc->matA = *matA;
    desc->matB = *matB;
    desc->matC = *matC;
    desc->matD = *matD;
    desc->computeType = computeType;
}

} // namespace

extern "C" {

aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType)
{
    if (handle == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (matmulDescr == nullptr || *matmulDescr != nullptr) {
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseStatus_t st = ValidateMatmulDescriptorParams(opA, opB, matA, matB, matC, matD, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    auto* desc = new (std::nothrow) aclsparseLtMatmulDescriptor();
    if (desc == nullptr) {
        return ACL_SPARSE_STATUS_ALLOC_FAILED;
    }

    FillMatmulDescriptor(desc, opA, opB, matA, matB, matC, matD, computeType);
    *matmulDescr = desc;
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr)
{
    if (matmulDescr == nullptr) {
        return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    }
    if (*matmulDescr == nullptr) {
        return ACL_SPARSE_STATUS_SUCCESS;
    }

    auto* desc = *matmulDescr;
    // matA/matB/matC/matD 为非所有权引用，Destroy 不销毁它们
    delete desc;
    *matmulDescr = nullptr;
    return ACL_SPARSE_STATUS_SUCCESS;
}

} // extern "C"
