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

#ifndef ACLSPARSE_CGTSV2_BACKEND_H_
#define ACLSPARSE_CGTSV2_BACKEND_H_

#include "cann_ops_sparse.h"

// The common Host owns the only two public C entry points. Each selected
// architecture supplies these internal workspace/dispatch functions.
bool Cgtsv2WorkspaceBytes(int m, int n, int ldb, size_t &bytes);
aclsparseStatus_t Cgtsv2Dispatch(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *b, int ldb, void *workspace);

#endif
