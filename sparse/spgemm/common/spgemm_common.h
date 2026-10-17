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
 * \file spgemm_common.h
 * \brief SpGEMM 公共层：dtype 分派、校验函数、arch22 上下文。
 */

#ifndef SPGEMM_COMMON_H
#define SPGEMM_COMMON_H

#include <cstdint>
#include <cstddef>
#include <vector>

#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"

enum SpgemmDtypeId : uint32_t {
    SPGEMM_DTYPE_FP32 = 0,
    SPGEMM_DTYPE_FP16 = 1,
    SPGEMM_DTYPE_BF16 = 2,
    SPGEMM_DTYPE_COMPLEX64 = 3,
};

enum SpgemmRowTemplate : uint8_t {
    SPGEMM_TPL_MERGE = 0,
    SPGEMM_TPL_CHUNK = 1,
};

/**
 * arch22 host 侧中间上下文。WorkEstimation 阶段创建，Compute/Copy 复用，
 * DestroyDescr 时释放。保存分核规划、D2H 缓存等 arch22 特有数据。
 */
struct SpgemmArch22Context {
    uint32_t cachedDtypeId = SPGEMM_DTYPE_FP32;
    bool bHasDuplicateColumns = false;
    int64_t maxRowProduct = 0;
    int64_t nnzCIn = 0;
    float cachedBetaRe = 0.0f;
    float cachedBetaIm = 0.0f;
    uint32_t blockDim = 0;
    std::vector<int64_t> rowProducts;
    std::vector<int32_t> binEdge;

    int64_t cRowOffsetsOffset = 0;
    int64_t cColIndicesOffset = 0;
    int64_t cValuesOffset = 0;
};

inline bool SpgemmDtypeFromAcl(aclDataType dt, uint32_t &dtypeId)
{
    switch (dt) {
        case ACL_FLOAT:      dtypeId = SPGEMM_DTYPE_FP32;      return true;
        case ACL_FLOAT16:    dtypeId = SPGEMM_DTYPE_FP16;      return true;
        case ACL_BF16:       dtypeId = SPGEMM_DTYPE_BF16;      return true;
        case ACL_COMPLEX64:  dtypeId = SPGEMM_DTYPE_COMPLEX64; return true;
        default:             return false;
    }
}

inline size_t SpgemmDtypeSize(uint32_t dtypeId)
{
    switch (dtypeId) {
        case SPGEMM_DTYPE_FP32:      return 4;
        case SPGEMM_DTYPE_FP16:      return 2;
        case SPGEMM_DTYPE_BF16:      return 2;
        case SPGEMM_DTYPE_COMPLEX64: return 8;
        default:                     return 4;
    }
}

inline bool SpgemmDtypeIsComplex(uint32_t dtypeId)
{
    return dtypeId == SPGEMM_DTYPE_COMPLEX64;
}

inline bool SpgemmAlgNeedsMemEstimate(aclsparseSpGEMMAlg_t alg)
{
    return alg == ACL_SPARSE_SPGEMM_ALG2 || alg == ACL_SPARSE_SPGEMM_ALG3;
}

aclsparseStatus_t ValidateSpgemmInputs(
    const aclsparseSpMatDescr *matA, const aclsparseSpMatDescr *matB,
    const aclsparseSpMatDescr *matC, aclsparseOperation_t opA,
    aclsparseOperation_t opB, aclDataType computeType, aclsparseSpGEMMAlg_t alg);

aclsparseStatus_t ValidateSpgemmStageConsistency(
    const aclsparseSpGEMMDescr *descr, const aclsparseSpMatDescr *matA,
    const aclsparseSpMatDescr *matB, aclDataType computeType,
    aclsparseOperation_t opA, aclsparseOperation_t opB, aclsparseSpGEMMAlg_t alg);

aclsparseStatus_t SpgemmReadScalar(const void *p, aclsparsePointerMode_t mode,
                                   uint32_t dtypeId, float &re, float &im);

#endif  // SPGEMM_COMMON_H
