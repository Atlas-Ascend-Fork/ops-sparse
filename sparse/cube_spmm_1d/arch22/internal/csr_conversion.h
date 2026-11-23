/**
 * Copyright (c) 2026 AscSpMM Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ascspmm {
// Host-only API. No CANN dependency. All indices are zero based.
struct Csr {
    int64_t m = 0, k = 0;
    std::vector<int64_t> rowptr, colidx;
    std::vector<double> values;
};
struct Stats {
    int64_t stored = 0, explicit_zeros = 0, canonical_nnz = 0;
    int64_t fp16_nnz = 0, underflow = 0, empty_blocks = 0;
};
struct VectorCsr {
    int32_t m = 0, k = 0, mpad = 0;
    int64_t pmax = 0;
    std::vector<int32_t> rowptr, colidx;
    std::vector<uint16_t> values; // IEEE binary16 bits; [P,16], vector first
    Stats stats;
};
// Rejects nonfinite values and |value| > 65504; RNE including subnormals.
uint16_t to_half(double value);
float from_half(uint16_t bits);
size_t checked_bytes(uint64_t count, size_t element_size);
// Stable row sort, FP64 left-to-right duplicate summation, exact-zero removal,
// then direct FP64 -> FP16 RNE. Output owns its memory.
VectorCsr convert(const Csr& input);
} // namespace ascspmm
