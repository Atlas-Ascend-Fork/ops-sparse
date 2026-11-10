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
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

inline std::string TrimCsvField(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

// Split one record on commas. A field wrapped in double quotes may contain
// commas, and "" inside it is an escaped quote. Same rule as the dedicated
// loader in test/sparse2dense/arch22, whose table quotes descriptions that
// contain commas. RFC 4180 allows nothing but a comma or the end of the record
// after a closing quote, so a tail such as `"a"x` is rejected instead of
// silently spliced into `ax`.
inline std::vector<std::string> SplitCsvRecord(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    bool atFieldStart = true;
    bool afterClosingQuote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted) {
            if (ch != '"') {
                field.push_back(ch);
            } else if (i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = false;
                afterClosingQuote = true;
            }
            continue;
        }
        if (ch == ',') {
            fields.push_back(field);
            field.clear();
            atFieldStart = true;
            afterClosingQuote = false;
        } else if (afterClosingQuote && ch != ' ' && ch != '\t' && ch != '\r') {
            // git stores the tables with LF, but a core.autocrlf checkout presents
            // them with CRLF and std::getline leaves that \r on the record, so a
            // trailing \r after the closing quote must not be a rejection.
            throw std::runtime_error("unexpected character after the closing quote of a quoted field");
        } else if (ch == '"' && atFieldStart) {
            field.clear();
            quoted = true;
        } else {
            field.push_back(ch);
            if (ch != ' ' && ch != '\t') atFieldStart = false;
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted field");
    }
    fields.push_back(field);
    return fields;
}

inline std::string CsvLocation(const std::string& csvPath, size_t lineNo) {
    return csvPath + ":" + std::to_string(lineNo) + ": ";
}

inline std::vector<std::string> ParseCsvHeader(const std::string& header, const std::string& csvPath,
                                               size_t lineNo) {
    std::vector<std::string> fields;
    try {
        fields = SplitCsvRecord(header);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(CsvLocation(csvPath, lineNo) + err.what());
    }
    std::vector<std::string> columns;
    columns.reserve(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
        const std::string name = TrimCsvField(fields[i]);
        if (name.empty()) {
            // Dropping the hole would shift every later value onto the wrong key.
            throw std::runtime_error(CsvLocation(csvPath, lineNo) + "empty column name at index " +
                                     std::to_string(i));
        }
        if (std::find(columns.begin(), columns.end(), name) != columns.end()) {
            throw std::runtime_error(CsvLocation(csvPath, lineNo) + "duplicate column name '" + name + "'");
        }
        columns.push_back(name);
    }
    return columns;
}

// These existing tables deliberately omit optional trailing parameters. Match
// both the file name and the complete optional suffix so a changed schema or a
// truncated required prefix cannot silently receive empty/default values.
inline bool CsvAllowsOmittedTrailingColumns(const std::string& csvPath,
                                           const std::vector<std::string>& columns, size_t fieldCount) {
    static const std::map<std::string, std::pair<size_t, std::vector<std::string>>> optionalSuffixes = {
        {"ltmatmul_test.csv", {23, {"bias_enabled", "bias_stride", "activation_type", "relu_upper_bound",
                                  "relu_threshold", "gelu_scaling", "num_batches", "batch_stride", "ld_sparse",
                                  "ld_dense", "dense_order", "cd_order", "ld_cd", "direct_feed"}}},
        {"prune_test.csv", {13, {"num_batches", "batch_stride", "ld", "in_place"}}},
        {"matmulSearch_test.csv", {39, {"ld_sparse", "ld_dense", "dense_order", "cd_order", "ld_cd"}}},
        {"spsv_test.csv", {22, {"rowptr_type", "colind_type", "force_permtype64"}}},
    };
    const size_t separator = csvPath.find_last_of("/\\");
    const std::string fileName = csvPath.substr(separator == std::string::npos ? 0 : separator + 1);
    const auto it = optionalSuffixes.find(fileName);
    if (it == optionalSuffixes.end()) return false;
    const size_t requiredCount = it->second.first;
    const auto& suffix = it->second.second;
    return columns.size() == requiredCount + suffix.size() && fieldCount >= requiredCount &&
           fieldCount <= columns.size() && std::equal(suffix.begin(), suffix.end(), columns.begin() + requiredCount);
}

// ReadMap: load CSV file into vector of csv_map rows. Reject malformed row
// widths; only the explicitly listed legacy optional suffixes may be omitted.
inline std::vector<csv_map> ReadMap(const std::string& csvPath) {
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        // Returning an empty vector would register zero GTest cases and exit 0.
        throw std::runtime_error("csv_loader: cannot open " + csvPath);
    }

    std::vector<std::string> columns;
    std::string line;
    size_t lineNo = 0;
    bool headerParsed = false;
    while (std::getline(ifs, line)) {
        ++lineNo;
        if (lineNo == 1U && line.size() >= 3U && static_cast<unsigned char>(line[0]) == 0xEFU &&
            static_cast<unsigned char>(line[1]) == 0xBBU && static_cast<unsigned char>(line[2]) == 0xBFU) {
            line.erase(0, 3);  // UTF-8 BOM would otherwise pollute the first column name
        }
        const std::string trimmed = TrimCsvField(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;  // preamble before the header
        columns = ParseCsvHeader(line, csvPath, lineNo);
        headerParsed = true;
        break;
    }
    if (!headerParsed) {
        throw std::runtime_error("csv_loader: missing header row in " + csvPath);
    }

    std::vector<csv_map> rows;
    while (std::getline(ifs, line)) {
        ++lineNo;
        const std::string trimmed = TrimCsvField(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        std::vector<std::string> fields;
        try {
            fields = SplitCsvRecord(line);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error(CsvLocation(csvPath, lineNo) + err.what());
        }
        if (fields.size() > columns.size() ||
            (fields.size() < columns.size() && !CsvAllowsOmittedTrailingColumns(csvPath, columns, fields.size()))) {
            throw std::runtime_error(CsvLocation(csvPath, lineNo) + "row has " + std::to_string(fields.size()) +
                                     " fields but the header has " + std::to_string(columns.size()));
        }

        csv_map row;
        for (size_t i = 0; i < fields.size(); ++i) {
            row[columns[i]] = TrimCsvField(fields[i]);
        }
        for (size_t i = fields.size(); i < columns.size(); ++i) {
            row[columns[i]] = "";
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        // A header without cases would otherwise register zero tests and pass.
        throw std::runtime_error(CsvLocation(csvPath, lineNo) + "missing data rows");
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

// Opt-in count check: a truncated or missing CSV must not silently reduce coverage.
template <typename ParamType>
std::vector<ParamType> GetCasesFromCsvChecked(const std::string& csvPath, size_t expectedCount)
{
    auto cases = GetCasesFromCsv<ParamType>(csvPath);
    if (cases.size() != expectedCount) {
        throw std::runtime_error("CSV case count mismatch for " + csvPath + ": expected " +
            std::to_string(expectedCount) + ", got " + std::to_string(cases.size()));
    }
    return cases;
}

}  // namespace sparse_test

#endif
