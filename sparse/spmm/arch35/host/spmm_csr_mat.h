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

#ifndef SPMM_CSR_MAT_H
#define SPMM_CSR_MAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "spmm_host.h"
#include "cann_ops_sparse.h"
#include "aclsparse_host_utils.h"
#include "aclsparse_descr_internal.h"

namespace spmm_host_internal { struct WsOffsets; }

class SpmmCsrMat {
public:
    SpmmCsrMat(aclsparseSpMatDescr *inner, int32_t blockDim)
        : matDesc(inner),
          blockDim(blockDim)
    {
        if (inner == nullptr) {
            printf("[SpmmCsrMat] inner descriptor is nullptr\n");
            abort();
        }
        m = static_cast<int64_t>(inner->rows);
        k = static_cast<int64_t>(inner->cols);
        nnz = static_cast<int64_t>(inner->nnz);
    }

    // Build the reorder table + row-bin edges into the device workspace,
    // and return them in host buffers so the caller can also fill TilingData.
    // identity=true skips the host-side scan and writes 0..m-1 + evenly split bins.
    uint32_t DoPreProcess(uint8_t *dWorkspaceBase,
                          int64_t reorderOffset,
                          int64_t binEdgeOffset,
                          bool identity);

    aclsparseStatus_t DoTransposePreProcess(uint8_t *workspace,
        const spmm_host_internal::WsOffsets &offsets);

    int32_t GetBinNum() const { return blockDim; }
    bool IsIdentityReorder() const { return identityReorder; }
    int32_t GetUniformDegree() const { return uniformDegree; }

private:
    aclsparseSpMatDescr *matDesc;
    int64_t m;
    int64_t k;
    int64_t nnz;
    int32_t blockDim;
    bool identityReorder = false;
    int32_t uniformDegree = -1;
};

#endif // SPMM_CSR_MAT_H
