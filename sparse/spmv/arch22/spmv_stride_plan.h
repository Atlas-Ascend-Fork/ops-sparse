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
#ifndef SPMV_STRIDE_PLAN_H
#define SPMV_STRIDE_PLAN_H

#include <cstddef>
#include <cstdint>
#include <limits>

// Independent byte-copy plan; the numerical kernel's tiling and descriptors stay unchanged.
struct SpmvStrideCopy {
    uint64_t length = 0;
    uint64_t strideBytes = 0;
    uint64_t offset = 0;
    uint32_t width = 0;
    uint32_t enabled = 0;
};
struct SpmvStridePlan {
    SpmvStrideCopy copies[5]; // CSR row offsets, columns, values, X, Y
    uint64_t vectorBytes = 0;
    uint64_t totalBytes = 0;
    uint32_t active = 0;
};

namespace spmv_stride {
inline bool DimensionsFit(uint64_t rows, uint64_t cols, uint64_t nnz, uint32_t indexBase) {
    return rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max() &&
           nnz <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) - indexBase;
}
inline bool Add(size_t a, size_t b, size_t &out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    out = a + b;
    return true;
}
inline bool Multiply(size_t a, size_t b, size_t &out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}
inline bool Align64(size_t value, size_t &out) {
    if (!Add(value, 63U, out)) return false;
    out &= ~size_t(63U);
    return true;
}
inline bool SpanFits(const void *ptr, uint64_t length, int64_t stride, size_t width) {
    if (stride <= 0 || width == 0) return false;
    if (length == 0) return true;
    size_t step = 0, span = 0;
    return Multiply(static_cast<size_t>(length <= 1 ? 1 : stride), width, step) &&
           Multiply(static_cast<size_t>(length - 1), step, span) && Add(span, width, span) &&
           reinterpret_cast<uintptr_t>(ptr) <= std::numeric_limits<uintptr_t>::max() - (span - 1);
}
// Allocate only arrays that need packing. All scratch starts on separate cache lines.
// totalBytes includes up to 63 bytes to align an arbitrarily offset caller buffer.
inline bool MakePlan(const uint64_t lengths[5], const int64_t strides[5], const uint32_t widths[5],
                     size_t vectorBytes, SpmvStridePlan &plan) {
    plan = SpmvStridePlan{};
    plan.vectorBytes = vectorBytes;
    bool active = false;
    for (unsigned i = 0; i < 5; ++i) {
        if (strides[i] <= 0 || widths[i] == 0) return false;
        active |= strides[i] != 1 && lengths[i] != 0;
    }
    if (!active) { plan.totalBytes = vectorBytes; return true; }
    size_t cursor = 0;
    if (!Align64(vectorBytes, cursor)) return false;
    for (unsigned i = 0; i < 5; ++i) {
        auto &copy = plan.copies[i];
        copy.length = lengths[i]; copy.width = widths[i];
        size_t step = 0;
        if (!Multiply(static_cast<size_t>(lengths[i] <= 1 ? 1 : strides[i]), widths[i], step)) return false;
        copy.strideBytes = step;
        if (strides[i] == 1 || lengths[i] == 0) continue;
        copy.enabled = 1; copy.offset = cursor;
        size_t bytes = 0;
        if (!Multiply(static_cast<size_t>(lengths[i]), widths[i], bytes) ||
            !Align64(bytes, bytes) || !Add(cursor, bytes, cursor)) return false;
    }
    if (!Add(cursor, 63U, cursor)) return false;
    plan.totalBytes = cursor; plan.active = 1;
    return true;
}
} // namespace spmv_stride
#endif
