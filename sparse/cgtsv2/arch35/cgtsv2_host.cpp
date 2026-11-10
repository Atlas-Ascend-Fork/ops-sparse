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

#include <algorithm>
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "cgtsv2_kernel.h"
#include "../cgtsv2_backend.h"

bool Cgtsv2WorkspaceBytes(int m, int n, int ldb, size_t& bytes)
{
    if (!cgtsv2::WorkspaceSize(m, n, ldb, bytes)) {
        OP_LOGE("aclsparseCgtsv2", "Invalid arch35 dimensions or size overflow: m=%d n=%d ldb=%d", m, n, ldb);
        return false;
    }
    return true;
}

aclsparseStatus_t Cgtsv2Dispatch(aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t* dl, const aclsparseComplex_t* d,
    const aclsparseComplex_t* du, aclsparseComplex_t* b, int ldb, void* workspace)
{
    aclrtStream stream = nullptr;
    const auto status = aclsparseGetStream(handle, &stream);
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        OP_LOGE("aclsparseCgtsv2", "aclsparseGetStream failed: status=%d m=%d n=%d ldb=%d",
            static_cast<int>(status), m, n, ldb);
        return status;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE("aclsparseCgtsv2", "GetAivCoreCount returned %u: m=%d n=%d ldb=%d", cores, m, n, ldb);
        return ACL_SPARSE_STATUS_INTERNAL_ERROR;
    }
    const uint32_t slots = std::min(static_cast<uint32_t>(n), cgtsv2::kMaxRhsSlots);
    const uint32_t blocks = std::min(cores, (slots + cgtsv2::kThreads - 1) / cgtsv2::kThreads);
    const uint32_t activeSlots = std::min(slots, blocks * cgtsv2::kThreads);
    const Cgtsv2TilingData tiling{m, n, ldb, blocks, activeSlots, cores};
    return cgtsv2_kernel_do(reinterpret_cast<uint8_t*>(const_cast<aclsparseComplex_t*>(dl)),
        reinterpret_cast<uint8_t*>(const_cast<aclsparseComplex_t*>(d)),
        reinterpret_cast<uint8_t*>(const_cast<aclsparseComplex_t*>(du)),
        reinterpret_cast<uint8_t*>(b), reinterpret_cast<uint8_t*>(workspace), tiling, stream);
}
