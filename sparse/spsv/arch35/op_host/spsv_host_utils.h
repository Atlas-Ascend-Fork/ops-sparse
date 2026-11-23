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

#ifndef SPSV_H_
#define SPSV_H_

#include <cstdint>
#include "cann_ops_sparse.h"
#include "aclsparse_descr_internal.h"
#include "aclsparse_spsv_descr.h"
#include "../shared/spsv_tiling_data.h" // kSimtMaxThreads canonical definition

namespace spsv {

// 351x (DAV_3510) GM addresses require 512-byte alignment for full bandwidth.
constexpr size_t kAlign = 512u;
// Sub-allocations within workspace use 64B alignment to reduce bloat for
// small matrices. Only the first offset (GM base) requires 512B alignment.
constexpr size_t kInternalAlign = 64u;

// Re-export the canonical kSimtMaxThreads (defined in spsv_tiling_data.h)
// into namespace spsv so host code can use spsv::kSimtMaxThreads.
using ::kSimtMaxThreads;

inline size_t AlignUp(size_t x) { return ((x + kAlign - 1u) / kAlign) * kAlign; }

inline size_t AlignUpInternal(size_t x) { return ((x + kInternalAlign - 1u) / kInternalAlign) * kInternalAlign; }

inline struct aclsparseContext* ToInternalHandle(aclsparseHandle_t handle)
{
    return reinterpret_cast<struct aclsparseContext*>(handle);
}

inline struct aclsparseSpMatDescr* ToMatInner(aclsparseConstSpMatDescr_t desc)
{
    return const_cast<struct aclsparseSpMatDescr*>(reinterpret_cast<const struct aclsparseSpMatDescr*>(desc));
}

inline struct aclsparseDnVecDescr* ToVecInner(aclsparseConstDnVecDescr_t desc)
{
    return const_cast<struct aclsparseDnVecDescr*>(reinterpret_cast<const struct aclsparseDnVecDescr*>(desc));
}

inline struct aclsparseDnVecDescr* ToVecInnerMut(aclsparseDnVecDescr_t desc)
{
    return reinterpret_cast<struct aclsparseDnVecDescr*>(desc);
}

inline int32_t FormatToInt(aclsparseFormat_t fmt)
{
    switch (fmt) {
        case ACL_SPARSE_FORMAT_CSR:
            return 0;
        case ACL_SPARSE_FORMAT_CSC:
            return 1;
        case ACL_SPARSE_FORMAT_COO:
            return 2;
        case ACL_SPARSE_FORMAT_SLICED_ELL:
            return 3;
        default:
            return -1;
    }
}

} // namespace spsv

#endif // SPSV_H_
