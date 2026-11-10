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

#include "cgtsv2_backend.h"
#include "cgtsv2_workspace.h"

#if !defined(OPS_SPARSE_CGTSV2_ARCH35) && !defined(OPS_SPARSE_CGTSV2_ARCH22)
bool Cgtsv2WorkspaceBytes(int m, int n, int ldb, size_t &bytes)
{
    return cgtsv2::WorkspaceBytes(m, n, ldb, bytes);
}

aclsparseStatus_t Cgtsv2Dispatch(aclsparseHandle_t, int, int,
    const aclsparseComplex_t *, const aclsparseComplex_t *,
    const aclsparseComplex_t *, aclsparseComplex_t *, int, void *)
{
    return ACL_SPARSE_STATUS_NOT_SUPPORTED;
}
#endif
