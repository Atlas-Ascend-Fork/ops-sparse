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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <set>
#include <string>
#include <vector>
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
struct Cgtsv2Param : public sparse_test::SparseTestParamBase {
    std::string caseName;
    std::string level;
    int m = 0;
    int n = 0;
    int ldb = 0;
    uint32_t seed = 0;
    std::string distribution;
    bool dominant = false;
    std::string expectedPath;
    std::string description;

    void fillCustom(const sparse_test::csv_map& row) override
    {
        caseName = sparse_test::parseString(row, "case_name");
        level = sparse_test::parseString(row, "level");
        m = sparse_test::parseInt(row, "m");
        n = sparse_test::parseInt(row, "n");
        ldb = sparse_test::parseInt(row, "ldb");
        const int64_t parsedSeed = sparse_test::parseStrictInt64(row, "seed");
        distribution = sparse_test::parseString(row, "distribution");
        dominant = sparse_test::parseBool(row, "dominant");
        expectedPath = sparse_test::parseString(row, "expected_path");
        description = sparse_test::parseString(row, "description");
        if (caseName.empty() || (level != "L0" && level != "L1") || m < 3 || n < 0 || ldb < m ||
            row.find("seed") == row.end() || row.at("seed").empty() ||
            parsedSeed < 0 || parsedSeed > std::numeric_limits<uint32_t>::max() ||
            (distribution != "uniform" && distribution != "normal") ||
            (expectedPath != "any" && expectedPath != "pcr" && expectedPath != "twofold")) {
            throw std::invalid_argument("Invalid Cgtsv2 CSV row: " + caseName);
        }
        seed = static_cast<uint32_t>(parsedSeed);
    }
    std::string caseId() const override { return caseName; }
};

inline void PrintTo(const Cgtsv2Param& p, std::ostream* stream)
{
    *stream << p.caseName << ": m=" << p.m << " n=" << p.n << " ldb=" << p.ldb
            << " seed=" << p.seed << " distribution=" << p.distribution
            << " dominant=" << p.dominant << " expected_path=" << p.expectedPath;
}

inline std::vector<Cgtsv2Param> LoadCases(const std::string& path)
{
    auto cases = sparse_test::GetCasesFromCsvChecked<Cgtsv2Param>(path, 97);
    size_t l0Cases = 0;
    std::set<std::string> caseIds;
    for (const auto& param : cases) {
        l0Cases += (param.level == "L0" ? 1U : 0U);
        if (!caseIds.insert(param.caseId()).second) {
            throw std::runtime_error("Duplicate Cgtsv2 CSV case: " + param.caseId() + " in " + path);
        }
    }
    if (l0Cases != 24) {
        throw std::runtime_error("Cgtsv2 CSV must contain L0=24 and L1=73: " + path +
            " (loaded L0=" + std::to_string(l0Cases) + ", L1=" +
            std::to_string(cases.size() - l0Cases) + ")");
    }
    return cases;
}
} // namespace cgtsv2_test
#endif
