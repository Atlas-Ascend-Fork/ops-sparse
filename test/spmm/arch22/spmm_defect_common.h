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

// Shared helpers and a small CSR/dense operand fixture for the arch22 SpMM
// defect-regression tests, kept in one header so the split test translation
// units do not duplicate code.

#pragma once

#include <acl/acl.h>
#include "cann_ops_sparse.h"
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <cstdint>

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_ACL_SPARSE(x)                                                                       \
    do {                                                                                          \
        aclsparseStatus_t __ret = x;                                                                \
        if (__ret != ACL_SPARSE_STATUS_SUCCESS) {                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclSparseError:" << __ret << std::endl; \
        }                                                                                         \
    } while (0);

#define CHECK_TEST(cond, msg)                                                \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("  [Fail] %s\n", msg);                                    \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// Device-side operands shared by the defect probes (small CSR matA1, dense B, C1).
struct D2Operands {
    int *dRowPtrA1 = NULL;
    int *dColIndA1 = NULL;
    float *dValA1 = NULL;
    float *dB = NULL;
    float *dC1 = NULL;
    int *dRowPtrA2 = NULL;
    int *dColIndA2 = NULL;
    float *dValA2 = NULL;
    float *dC2 = NULL;
    aclsparseSpMatDescr_t matA1 = NULL;
    aclsparseSpMatDescr_t matA2 = NULL;
    aclsparseDnMatDescr_t matB = NULL;
    aclsparseDnMatDescr_t matC1 = NULL;
    aclsparseDnMatDescr_t matC2 = NULL;
};

static const int64_t kD2M1 = 4;
static const int64_t kD2M2 = 64;
static const int64_t kD2K = 3;
static const int64_t kD2N = 2;
static const float kD2Expected1[8] = {3.0f, 2.0f, 0.0f, 3.0f, 4.0f, 0.0f, 5.0f, 5.0f};

// Small CSR fixture (matA1: 4x3, 5 nnz) + dense B (3x2)，D2/D6 共用，避免分拆的测试
// 翻译单元重复同一份字面量数据（codecheck 重复代码）。
static const int kD2RowPtrA1[] = {0, 2, 3, 4, 5};
static const int kD2ColIndA1[] = {0, 2, 1, 0, 2};
static const float kD2ValA1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
static const float kD2B[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

// Builds matA1 (the small sparse operand), the shared dense B, and C1.
static inline int d2_build_small_operands(D2Operands &ops)
{
    const int64_t nnz1 = 5;
    CHECK_ACL(aclrtMalloc((void **)&ops.dRowPtrA1, sizeof(kD2RowPtrA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dColIndA1, sizeof(kD2ColIndA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dValA1, sizeof(kD2ValA1), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dB, sizeof(kD2B), ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMalloc((void **)&ops.dC1, static_cast<size_t>(kD2M1) * kD2N * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST))
    CHECK_ACL(aclrtMemcpy(ops.dRowPtrA1, sizeof(kD2RowPtrA1), kD2RowPtrA1, sizeof(kD2RowPtrA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dColIndA1, sizeof(kD2ColIndA1), kD2ColIndA1, sizeof(kD2ColIndA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dValA1, sizeof(kD2ValA1), kD2ValA1, sizeof(kD2ValA1),
        ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL(aclrtMemcpy(ops.dB, sizeof(kD2B), kD2B, sizeof(kD2B), ACL_MEMCPY_HOST_TO_DEVICE))
    CHECK_ACL_SPARSE(aclsparseCreateCsr(&ops.matA1, kD2M1, kD2K, nnz1,
        ops.dRowPtrA1, ops.dColIndA1, ops.dValA1,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&ops.matB, kD2K, kD2N, kD2N, ops.dB, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW))
    CHECK_ACL_SPARSE(aclsparseCreateDnMat(&ops.matC1, kD2M1, kD2N, kD2N, ops.dC1, ACL_FLOAT,
        ACL_SPARSE_ORDER_ROW))
    return 0;
}

// 缺陷回归入口（定义于 spmm_defect_extra_test.cpp），供 spmm_test.cpp 的 main 调用。
// 统一在头文件声明，避免各翻译单元用 extern 直接引用外部函数（G.EXP.05-CPP）。
int run_defect_regression_tests_extra(aclrtStream stream);
int run_precision_regression_tests(aclrtStream stream);
