/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_SPSV_HOST_UTILS_H_
#define SPSV_ARCH22_SPSV_HOST_UTILS_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#include "cann_ops_sparse.h"
#include "acl/acl_rt.h"
#include "acl/error_codes/rt_error_codes.h"

namespace spsv_common {
inline aclsparseStatus_t ValidateOperation(int op)
{
    return op == ACL_SPARSE_OP_NON_TRANSPOSE || op == ACL_SPARSE_OP_TRANSPOSE ||
        op == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ? ACL_SPARSE_STATUS_SUCCESS :
        ACL_SPARSE_STATUS_INVALID_VALUE;
}
inline bool IsValueTypeSupported(int type) { return type == ACL_FLOAT || type == ACL_COMPLEX64; }
}

namespace spsv_memory {
inline bool Contains(uintptr_t address, size_t requiredBytes, uintptr_t start, size_t bytes)
{
    return address >= start && address - start < bytes && requiredBytes <= bytes - (address - start);
}

inline aclError ValidateRangeResult(const void *pointer, size_t requiredBytes, aclError queryResult,
    const void *base, size_t bytes)
{
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const size_t span = requiredBytes == 0 ? 1 : requiredBytes;
    if (pointer == nullptr || span - 1 > std::numeric_limits<uintptr_t>::max() - address)
        return ACL_ERROR_RT_PARAM_INVALID;
    // Unsupported capability is distinct from an invalid pointer. The caller
    // retains its public location/device checks and decides whether to proceed.
    if (queryResult != ACL_SUCCESS) return queryResult;
    const auto start = reinterpret_cast<uintptr_t>(base);
    if (base == nullptr || bytes == 0 || bytes - 1 > std::numeric_limits<uintptr_t>::max() - start)
        return ACL_ERROR_RT_PARAM_INVALID;
    return Contains(address, span, start, bytes) ? ACL_SUCCESS : ACL_ERROR_RT_PARAM_INVALID;
}

inline aclError ValidateRange(const void *pointer, size_t requiredBytes)
{
    void *base = nullptr;
    size_t bytes = 0;
    const auto result = aclrtMemGetAddressRange(const_cast<void *>(pointer), &base, &bytes);
    return ValidateRangeResult(pointer, requiredBytes, result, base, bytes);
}
} // namespace spsv_memory

#endif // SPSV_ARCH22_SPSV_HOST_UTILS_H_
