/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_CSV_LOADER_H_
#define TEST_FRAME_CSV_LOADER_H_

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sparse_test {

// csv_map: column_name -> value (string)
using csv_map = std::map<std::string, std::string>;

// Parse helpers
inline bool parseBool(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) return false;
    return (it->second == "1" || it->second == "true" || it->second == "yes");
}

inline int parseInt(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) return 0;
    return std::atoi(it->second.c_str());
}

inline int64_t parseInt64(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) return 0;
    return static_cast<int64_t>(std::strtoll(it->second.c_str(), nullptr, 10));
}

// Strict int64 parsing: a missing column/empty value returns 0
// (unspecified); a non-empty but invalid value (no valid characters,
// leftover characters, or overflow) returns the INT64_MIN error sentinel,
// preventing silent 0 for illegal input from being confused with a legal 0.
// Coexists with the lenient parseInt64 (strtoll direct conversion, no
// leftover-character validation): negative cases that must distinguish
// "unspecified" from "illegal input" (e.g. descriptor rows/cols/ld
// out-of-range injection) use the strict version, while ordinary
// missing-means-0 parameter parsing keeps the lenient version.
inline int64_t parseStrictInt64(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end() || it->second.empty()) { return 0; }
    const std::string& s = it->second;
    errno = 0;
    char* endptr = nullptr;
    const long long val = std::strtoll(s.c_str(), &endptr, 10);
    if (endptr == s.c_str() || *endptr != '\0' || errno == ERANGE) {
        return INT64_MIN; // error sentinel for illegal input
    }
    return static_cast<int64_t>(val);
}

inline double parseDouble(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) return 0.0;
    return std::atof(it->second.c_str());
}

inline float parseFloat(const csv_map& row, const std::string& key) {
    return static_cast<float>(parseDouble(row, key));
}

inline std::string parseString(const csv_map& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) return "";
    return it->second;
}

// ReadMap: load CSV file into vector of csv_map rows
inline std::vector<csv_map> ReadMap(const std::string& csvPath) {
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        std::cerr << "ReadMap: cannot open " << csvPath << std::endl;
        return {};
    }

    std::vector<csv_map> rows;
    std::string header;
    std::getline(ifs, header);

    // Parse header columns
    std::vector<std::string> columns;
    {
        std::stringstream ss(header);
        std::string col;
        while (std::getline(ss, col, ',')) {
            // Strip whitespace
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            if (!col.empty()) columns.push_back(col);
        }
    }

    // Parse data rows
    std::string line;
    while (std::getline(ifs, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        // Trim trailing whitespace
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        csv_map row;
        std::stringstream ss(line);
        std::string val;
        size_t colIdx = 0;
        while (std::getline(ss, val, ',') && colIdx < columns.size()) {
            val.erase(0, val.find_first_not_of(" \t"));
            val.erase(val.find_last_not_of(" \t") + 1);
            row[columns[colIdx]] = val;
            colIdx++;
        }
        // Pad remaining columns with empty string
        while (colIdx < columns.size()) {
            row[columns[colIdx]] = "";
            colIdx++;
        }
        rows.push_back(row);
    }

    return rows;
}

// Base class for test parameters loaded from CSV
class SparseTestParamBase {
public:
    virtual ~SparseTestParamBase() = default;

    // Override in subclass to parse custom columns from csv_map
    virtual void fillCustom(const csv_map& row) = 0;

    // Override in subclass to return a unique case identifier
    virtual std::string caseId() const = 0;
};

// Template function: load CSV and convert to vector of ParamType
template <typename ParamType>
std::vector<ParamType> GetCasesFromCsv(const std::string& csvPath) {
    auto rawRows = ReadMap(csvPath);
    std::vector<ParamType> cases;
    cases.reserve(rawRows.size());
    for (const auto& row : rawRows) {
        ParamType p;
        p.fillCustom(row);
        cases.push_back(std::move(p));
    }
    return cases;
}

}  // namespace sparse_test

#endif
