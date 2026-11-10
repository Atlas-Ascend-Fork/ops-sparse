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

#ifndef ACLSPARSE_CGTSV2_WORKSPACE_H_
#define ACLSPARSE_CGTSV2_WORKSPACE_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cgtsv2 {
constexpr size_t kWorkspaceAlignment = 128;

inline bool CheckedMultiply(size_t a, size_t b, size_t &out)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

inline bool WorkspaceBytes(int m, int n, int ldb, size_t &bytes)
{
    bytes = 0;
    if (m < 3 || n < 0 || ldb < m) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    size_t bElements = 0;
    size_t bBytes = 0;
    size_t wsElements = 0;
    size_t rawBytes = 0;
    if (!CheckedMultiply(static_cast<size_t>(ldb), static_cast<size_t>(n), bElements) ||
        !CheckedMultiply(bElements, 8, bBytes) ||
        !CheckedMultiply(static_cast<size_t>(m), static_cast<size_t>(n), wsElements) ||
        !CheckedMultiply(wsElements, 80, rawBytes)) {
        return false;
    }
    const size_t maxOffset = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (bBytes > maxOffset || rawBytes > maxOffset - kWorkspaceAlignment ||
        rawBytes > std::numeric_limits<size_t>::max() - (2 * kWorkspaceAlignment - 1)) {
        return false;
    }
    bytes = kWorkspaceAlignment + ((rawBytes + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1));
    return true;
}
} // namespace cgtsv2
#endif
