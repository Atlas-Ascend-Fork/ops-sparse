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
#ifndef TEST_CGTSV2_PARAM_H_
#define TEST_CGTSV2_PARAM_H_

#include "csv_loader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace cgtsv2_test {
inline int ParseDimension(const sparse_test::csv_map &row, const std::string &key)
{
    const auto value = sparse_test::parseString(row, key);
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("Invalid cgtsv2 CSV integer: " + key);
    }
    return result;
}

struct Param : sparse_test::SparseTestParamBase {
    std::string name;
    std::string kind;
    int m = 0;
    int n = 0;
    int ldb = 0;
    void fillCustom(const sparse_test::csv_map &row) override
    {
        name = sparse_test::parseString(row, "case_name");
        kind = sparse_test::parseString(row, "kind");
        m = ParseDimension(row, "m");
        n = ParseDimension(row, "n");
        ldb = ParseDimension(row, "ldb");
        constexpr std::array<std::string_view, 13> validKinds = {"dominant", "pivot", "zero_pivot", "modulus_tie",
            "imaginary", "scaled_large", "scaled_tiny", "jacobi_boundary_126", "jacobi_boundary_135",
            "jacobi_boundary_136", "jacobi_boundary_150", "jacobi_boundary_200", "jacobi_boundary_201"};
        if (m < 3 || n < 0 || ldb < m ||
            std::find(validKinds.begin(), validKinds.end(), kind) == validKinds.end()) {
            throw std::invalid_argument("Invalid cgtsv2 CSV dimensions or kind: " + name);
        }
    }
    std::string caseId() const override { return name; }
};
} // namespace cgtsv2_test
#endif
