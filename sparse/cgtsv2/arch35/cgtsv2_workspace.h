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

#ifndef CGTSV2_WORKSPACE_H_
#define CGTSV2_WORKSPACE_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cgtsv2 {
constexpr uint32_t kThreads = 32;
constexpr uint32_t kParallelThreads = 256;
constexpr uint32_t kMaxRhsSlots = 128;
constexpr size_t kAlignment = 128;
constexpr size_t kComplexBytes = 8;
// Four extended-range twofold-complex arrays (24 bytes per value):
// three upper diagonals and the transformed RHS.
constexpr size_t kFactorArrays = 12;

// Grid-stride RHS processing reuses bounded workspace slots, independent of n.
inline bool WorkspaceSize(int m, int n, int ldb, size_t& bytes)
{
    bytes = 0;
    if (m < 3 || n < 0 || ldb < m) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    const size_t limit = std::numeric_limits<size_t>::max();
    const size_t rows = static_cast<size_t>(m);
    const size_t columns = static_cast<size_t>(n);
    if (columns > limit / kComplexBytes / static_cast<size_t>(ldb)) {
        return false;
    }
    const size_t slots = columns < kMaxRhsSlots ? columns : kMaxRhsSlots;
    if (rows > (limit - (2 * kAlignment - 1)) / kComplexBytes / kFactorArrays / slots) {
        return false;
    }
    bytes = rows * slots * kFactorArrays * kComplexBytes;
    bytes = ((bytes + kAlignment - 1) & ~(kAlignment - 1)) + kAlignment;
    return true;
}
} // namespace cgtsv2
#endif
