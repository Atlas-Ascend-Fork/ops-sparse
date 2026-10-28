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

#ifndef CUBE_SPMM_PREPROCESS_H_
#define CUBE_SPMM_PREPROCESS_H_

#include <cstdint>
#include "cann_ops_sparse.h"

/**
 * @brief Internal helper: COO -> BCSR conversion on the host, using device COO
 * inputs. It copies the COO arrays to host, builds BCSR, allocates device
 * output buffers and copies the results back.
 *
 * The caller must free the returned device buffers with aclrtFree.
 */
aclsparseStatus_t CubeSpmmBuildBcsrOnDevice(
    int64_t M, int64_t K, int64_t nnz,
    const void *cooRowsDev, const void *cooColsDev, const void *cooValsDev,
    int64_t blockM, int64_t blockK, int32_t numCores,
    void **rwPtrOut, void **colRefOut, void **valsOut, void **coreInfoOut,
    int64_t *rwPtrElems, int64_t *colRefElems, int64_t *valsElems, int64_t *coreInfoElems,
    bool *needsOutputPreclearOut);

#endif  // CUBE_SPMM_PREPROCESS_H_
