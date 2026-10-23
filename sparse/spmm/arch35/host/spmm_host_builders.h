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

#ifndef SPMM_HOST_BUILDERS_H
#define SPMM_HOST_BUILDERS_H

#include <stdint.h>

#include "aclsparse_descr_internal.h"
#include "cann_ops_sparse.h"
#include "spmm_host_internal.h"

namespace spmm_host_builders {

aclsparseStatus_t BuildComplexRegBaseData(
    aclsparseSpMatDescr *matA, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups);

aclsparseStatus_t BuildComplexCubePrepackedData(
    aclsparseSpMatDescr *matA, const aclsparseDnMatDescr *matB, int32_t n,
    uint8_t *workspace, const spmm_host_internal::WsOffsets &offsets,
    bool candidate, bool *eligible, int32_t *selectedRows,
    int32_t *selectedGroups, int32_t *selectedFallbackCount);

aclsparseStatus_t BuildCubePrepackedData(
    aclsparseSpMatDescr *matA, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount);

aclsparseStatus_t BuildFp32RegBaseData(
    aclsparseSpMatDescr *matA, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups);

aclsparseStatus_t BuildFp32Degree2Data(
    aclsparseSpMatDescr *matA, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount);

aclsparseStatus_t BuildFp32Degree4Data(
    aclsparseSpMatDescr *matA, int32_t n, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups,
    int32_t *selectedFallbackCount);

aclsparseStatus_t BuildFp32PackedData(
    aclsparseSpMatDescr *matA, int32_t n, uint8_t *workspace,
    const spmm_host_internal::WsOffsets &offsets, bool candidate,
    bool *eligible, int32_t *selectedRows, int32_t *selectedGroups);

} // namespace spmm_host_builders

#endif // SPMM_HOST_BUILDERS_H
