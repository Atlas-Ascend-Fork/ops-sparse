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
 * \file aclsparse_host_utils.h
 * \brief Host-side utilities for ops-sparse.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "cann_ops_sparse.h"

/// 校验 Legacy MatDescr（aclsparseMatDescr_t）三连约束：非空、
/// type 必须为 GENERAL、indexBase 必须为 ZERO 或 ONE。
/// 此前在 csr2gebsr.h / csrgeam2.h（带 tag 改名副本）与 nnz host
/// （static 无 tag 副本）三处逐字重复，收敛为本头文件单点定义。
inline aclsparseStatus_t AclsparseValidateMatDescr(const char *apiTag,
                                                   const char *paramName,
                                                   const aclsparseMatDescr_t descr)
{
    if (descr == nullptr) {
        OP_LOGE(apiTag, "%s is nullptr", paramName);
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    aclsparseMatrixType_t matType = aclsparseGetMatType(descr);
    if (matType != ACL_SPARSE_MATRIX_TYPE_GENERAL) {
        OP_LOGE(apiTag, "%s: unsupported matrix type %d",
                paramName, static_cast<int>(matType));
        return ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    aclsparseIndexBase_t indexBase = aclsparseGetMatIndexBase(descr);
    if (indexBase != ACL_SPARSE_INDEX_BASE_ZERO &&
        indexBase != ACL_SPARSE_INDEX_BASE_ONE) {
        OP_LOGE(apiTag, "%s: invalid indexBase %d",
                paramName, static_cast<int>(indexBase));
        return ACL_SPARSE_STATUS_INVALID_VALUE;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

#ifndef CHECK_RET
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#endif

#ifndef CHECK_ACL
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            OP_LOGE("aclsparse", "aclError: %d at %s:%d", __ret, __FILE__, __LINE__);       \
            return ACL_SPARSE_STATUS_INTERNAL_ERROR;                                         \
        }                                                                                   \
    } while (0)
#endif

template <typename R = uint32_t, typename T1, typename T2>
static inline R CeilDiv(T1 a, T2 b)
{
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
        "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_unsigned<R>::value,
        "CeilDiv return type R must be unsigned to ensure safe overflow detection");
    auto ra = static_cast<R>(a);
    auto rb = static_cast<R>(b);
    if (rb == 0) {
        return std::numeric_limits<R>::max();
    }
    if (ra + rb - 1 < ra) {
        return std::numeric_limits<R>::max();
    }
    return (ra + rb - 1) / rb;
}

static inline uint32_t GetAivCoreCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclsparse", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAiv();
}

static inline uint32_t GetAicCoreCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclsparse", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    return platform->GetCoreNumAic();
}

static inline uint64_t GetUbSize()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclsparse", "PlatformAscendCManager is null, return 0");
        return 0;
    }
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE("aclsparse", "GetCoreMemSize(UB) returned 0");
    }
    return ubSize;
}
