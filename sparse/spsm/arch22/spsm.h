/**
 * Copyright (c) 2026 liutaw.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSM_ARCH22_H
#define SPSM_ARCH22_H
#include "cann_ops_sparse.h"
#include "spsm_plan.h"

struct SpsmA2Binding {
    aclsparseHandle_t handle{nullptr};
    aclsparseConstSpMatDescr_t matA{nullptr};
    aclsparseConstDnMatDescr_t matB{nullptr};
    aclsparseDnMatDescr_t matC{nullptr};
    void *first{nullptr}, *second{nullptr}, *aValues{nullptr};
    void *bValues{nullptr}, *cValues{nullptr};
    int64_t bRows{0}, bCols{0}, bLd{0}, cRows{0}, cCols{0}, cLd{0};
    aclsparseOrder_t bOrder{}, cOrder{};
    aclrtStream stream{nullptr};
    int32_t device{-1};
    spsm_a2::PlanConfig config{};
    spsm_a2::WorkspaceLayout layout{};
};

// Device storage is caller-owned. This object stores only fixed-size metadata.
struct aclsparseSpSMDescr {
    bool analyzed{false};
    void *workspace{nullptr};
    SpsmA2Binding binding{};
    aclsparseSpSMDescr *next{nullptr};
};
#endif
