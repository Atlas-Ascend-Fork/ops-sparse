/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef TEST_DENSETOSPARSE_GOLDEN_H_
#define TEST_DENSETOSPARSE_GOLDEN_H_

#ifdef SPARSE_TEST_USE_EIGEN
#include <Eigen/Sparse>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparse_test {

enum class DenseSparseFormat { CSR, CSC, COO, BLOCKED_ELL };

struct DenseToSparseGoldenResult {
  int64_t nnz = 0;        // Actual nnz; not used for Blocked-ELL.
  int64_t genericNnz = 0; // Blocked-ELL descriptor metadata only.
  std::vector<int64_t> offsets;
  std::vector<int64_t> indices0;
  std::vector<int64_t> indices1;
  std::vector<int64_t> ellPattern;
  std::vector<uint8_t> values;
};

inline bool DenseToSparseCheckedMul(uint64_t a, uint64_t b, uint64_t *out) {
  if (out == nullptr ||
      (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

inline bool DenseToSparseIsZero(const uint8_t *value, size_t width,
                                bool integer) {
  if (integer)
    return value[0] == 0;
  if (width == 2) {
    uint16_t bits = 0;
    std::copy_n(value, sizeof(bits), reinterpret_cast<uint8_t *>(&bits));
    return (bits & 0x7fffu) == 0; // FP16 and BF16 share signed-zero encoding.
  }
  if (width == 4) {
    uint32_t bits = 0;
    std::copy_n(value, sizeof(bits), reinterpret_cast<uint8_t *>(&bits));
    return (bits & 0x7fffffffu) == 0;
  }
  if (width == 8) {
    // complex64: zero only when both FP32 components have zero magnitude bits.
    uint64_t bits = 0;
    std::copy_n(value, sizeof(bits), reinterpret_cast<uint8_t *>(&bits));
    return ((static_cast<uint32_t>(bits) & 0x7fffffffu) |
            (static_cast<uint32_t>(bits >> 32) & 0x7fffffffu)) == 0;
  }
  throw std::invalid_argument("unsupported floating value width");
}

inline int64_t DenseToSparseCeilDiv(int64_t value, int64_t divisor) {
  if (divisor < 0) {
    return 0;
  }
  if (divisor == 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor;
}

// Blocked-ELL capacity on the padded block grid; equals rows*ellCols whenever
// rows and ellCols are multiples of blockSize.
inline int64_t DenseToSparseBellValueCount(int64_t rows, int64_t blockSize,
                                           int64_t ellCols) {
  return DenseToSparseCeilDiv(rows, blockSize) *
         DenseToSparseCeilDiv(ellCols, blockSize) * blockSize * blockSize;
}

inline int64_t DenseToSparseBellPatternCount(int64_t rows, int64_t blockSize,
                                             int64_t ellCols) {
  return DenseToSparseCeilDiv(rows, blockSize) *
         DenseToSparseCeilDiv(ellCols, blockSize);
}

inline size_t DenseToSparsePhysicalIndex(int64_t row, int64_t col, int64_t ld,
                                         bool rowMajor) {
  return static_cast<size_t>(rowMajor ? row * ld + col : col * ld + row);
}

inline void DenseToSparseAppendValue(std::vector<uint8_t> *out,
                                     const std::vector<uint8_t> &dense,
                                     size_t element, size_t width) {
  const size_t begin = element * width;
  if (begin > dense.size() || width > dense.size() - begin) {
    throw std::out_of_range("dense physical buffer too small");
  }
  out->insert(out->end(), dense.begin() + begin, dense.begin() + begin + width);
}

struct DenseToSparseGoldenArgs {
  DenseSparseFormat format;
  int64_t m;
  int64_t n;
  int64_t ld;
  bool rowMajor;
  int64_t base;
  size_t valueWidth;
  bool integer;
  const std::vector<uint8_t> &dense;
};

inline void
DenseToSparseValidateGoldenArgs(const DenseToSparseGoldenArgs &args) {
  if (args.m < 0 || args.n < 0 || args.ld < 0 ||
      (args.base != 0 && args.base != 1)) {
    throw std::invalid_argument("invalid dense-to-sparse shape/base");
  }
  const int64_t minLd = args.rowMajor ? args.n : args.m;
  if ((args.m != 0 && args.n != 0 && args.ld < minLd) || args.valueWidth == 0) {
    throw std::invalid_argument("invalid dense-to-sparse ld/value width");
  }
  uint64_t physical = 0;
  if (!DenseToSparseCheckedMul(
          static_cast<uint64_t>(args.rowMajor ? args.m : args.n),
          static_cast<uint64_t>(args.ld), &physical) ||
      physical > std::numeric_limits<size_t>::max() / args.valueWidth ||
      args.dense.size() < static_cast<size_t>(physical) * args.valueWidth) {
    throw std::overflow_error("dense physical span overflow");
  }
}

inline void DenseToSparseCopyBellValue(const DenseToSparseGoldenArgs &args,
                                       int64_t blockSize, int64_t slots,
                                       int64_t blockRow, int64_t slot,
                                       int64_t blockCol, int64_t innerCol,
                                       int64_t innerRow,
                                       DenseToSparseGoldenResult *result) {
  const int64_t row = blockRow * blockSize + innerRow;
  const int64_t col = blockCol * blockSize + innerCol;
  const size_t dstElement = static_cast<size_t>(
      ((blockRow * slots + slot) * blockSize + innerCol) * blockSize +
      innerRow);
  const size_t srcElement =
      DenseToSparsePhysicalIndex(row, col, args.ld, args.rowMajor);
  std::copy_n(args.dense.begin() + srcElement * args.valueWidth,
              args.valueWidth,
              result->values.begin() + dstElement * args.valueWidth);
}

inline bool DenseToSparseBellBlockColumnValid(int64_t encoded,
                                               int64_t blockCol, int64_t n,
                                               int64_t safeBlockSize) {
  const int64_t nonzeroBlockSize =
      safeBlockSize > 0 ? safeBlockSize : 1;
  return encoded != -1 && blockCol >= 0 &&
         blockCol < DenseToSparseCeilDiv(n, nonzeroBlockSize);
}

/// 把一个非零块的有效区域搬进它在 Blocked-ELL 里的 slot。超出矩阵的逻辑元素保持
/// 结果缓冲初始化时的正零。
inline void DenseToSparseFillBellBlock(const DenseToSparseGoldenArgs &args,
                                       int64_t blockSize, int64_t slots,
                                       int64_t blockRow, int64_t slot,
                                       int64_t blockCol,
                                       DenseToSparseGoldenResult *result) {
  for (int64_t innerCol = 0; innerCol < blockSize; ++innerCol) {
    for (int64_t innerRow = 0; innerRow < blockSize; ++innerRow) {
      if (blockRow * blockSize + innerRow >= args.m ||
          blockCol * blockSize + innerCol >= args.n) {
        continue;
      }
      DenseToSparseCopyBellValue(args, blockSize, slots, blockRow, slot,
                                 blockCol, innerCol, innerRow, result);
    }
  }
}

inline DenseToSparseGoldenResult
DenseToSparseBellGolden(const DenseToSparseGoldenArgs &args, int64_t blockSize,
                        int64_t ellCols,
                        const std::vector<int64_t> &ellPattern) {
  if (blockSize <= 0) {
    throw std::invalid_argument("invalid Blocked-ELL geometry");
  }
  const int64_t safeBlockSize = blockSize > 0 ? blockSize : 1;
  if (ellCols < 0 ||
      DenseToSparseCeilDiv(ellCols, safeBlockSize) >
          DenseToSparseCeilDiv(args.n, safeBlockSize)) {
    throw std::invalid_argument("invalid Blocked-ELL geometry");
  }
  const uint64_t patternCount = static_cast<uint64_t>(
      DenseToSparseBellPatternCount(args.m, safeBlockSize, ellCols));
  const uint64_t valueCount = static_cast<uint64_t>(
      DenseToSparseBellValueCount(args.m, safeBlockSize, ellCols));
  if (ellPattern.size() != patternCount ||
      valueCount > std::numeric_limits<size_t>::max() / args.valueWidth) {
    throw std::overflow_error("Blocked-ELL capacity overflow/mismatch");
  }
  DenseToSparseGoldenResult result;
  result.genericNnz = static_cast<int64_t>(valueCount);
  result.ellPattern = ellPattern;
  result.values.assign(static_cast<size_t>(valueCount) * args.valueWidth, 0);
  const int64_t slots = DenseToSparseCeilDiv(ellCols, safeBlockSize);
  const int64_t blockRows = DenseToSparseCeilDiv(args.m, safeBlockSize);
  for (int64_t blockRow = 0; blockRow < blockRows; ++blockRow) {
    for (int64_t slot = 0; slot < slots; ++slot) {
      const int64_t encoded =
          ellPattern[static_cast<size_t>(blockRow * slots + slot)];
      const int64_t blockCol = encoded - args.base;
      if (!DenseToSparseBellBlockColumnValid(encoded, blockCol, args.n,
                                             safeBlockSize)) {
        continue;
      }
      DenseToSparseFillBellBlock(args, safeBlockSize, slots, blockRow, slot,
                                 blockCol, &result);
    }
  }
  return result;
}

inline void
DenseToSparseScanCsc(const DenseToSparseGoldenArgs &args,
                     std::vector<std::pair<int64_t, int64_t>> *coordinates,
                     DenseToSparseGoldenResult *result) {
  result->offsets.push_back(args.base);
  for (int64_t col = 0; col < args.n; ++col) {
    for (int64_t row = 0; row < args.m; ++row) {
      const size_t element =
          DenseToSparsePhysicalIndex(row, col, args.ld, args.rowMajor);
      if (DenseToSparseIsZero(args.dense.data() + element * args.valueWidth,
                              args.valueWidth, args.integer)) {
        continue;
      }
      coordinates->emplace_back(row, col);
      result->indices0.push_back(row + args.base);
      DenseToSparseAppendValue(&result->values, args.dense, element,
                               args.valueWidth);
    }
    result->offsets.push_back(static_cast<int64_t>(coordinates->size()) +
                              args.base);
  }
}

inline void
DenseToSparseScanRows(const DenseToSparseGoldenArgs &args,
                      std::vector<std::pair<int64_t, int64_t>> *coordinates,
                      DenseToSparseGoldenResult *result) {
  if (args.format == DenseSparseFormat::CSR) {
    result->offsets.push_back(args.base);
  }
  for (int64_t row = 0; row < args.m; ++row) {
    for (int64_t col = 0; col < args.n; ++col) {
      const size_t element =
          DenseToSparsePhysicalIndex(row, col, args.ld, args.rowMajor);
      if (DenseToSparseIsZero(args.dense.data() + element * args.valueWidth,
                              args.valueWidth, args.integer)) {
        continue;
      }
      coordinates->emplace_back(row, col);
      DenseToSparseAppendValue(&result->values, args.dense, element,
                               args.valueWidth);
      result->indices0.push_back(
          (args.format == DenseSparseFormat::CSR ? col : row) + args.base);
      if (args.format == DenseSparseFormat::COO) {
        result->indices1.push_back(col + args.base);
      }
    }
    if (args.format == DenseSparseFormat::CSR) {
      result->offsets.push_back(static_cast<int64_t>(coordinates->size()) +
                                args.base);
    }
  }
}

inline void DenseToSparseValidateStructure(
    int64_t m, int64_t n,
    const std::vector<std::pair<int64_t, int64_t>> &coordinates) {
#ifdef SPARSE_TEST_USE_EIGEN
  using Triplet = Eigen::Triplet<double, int64_t>;
  std::vector<Triplet> triplets;
  triplets.reserve(coordinates.size());
  for (size_t i = 0; i < coordinates.size(); ++i) {
    triplets.emplace_back(coordinates[i].first, coordinates[i].second,
                          static_cast<double>(i + 1));
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> structure(m, n);
  structure.setFromTriplets(triplets.begin(), triplets.end());
  if (static_cast<size_t>(structure.nonZeros()) != coordinates.size()) {
    throw std::logic_error("Eigen structural nnz cross-check failed");
  }
#else
  const std::set<std::pair<int64_t, int64_t>> unique(coordinates.begin(),
                                                     coordinates.end());
  if (unique.size() != coordinates.size()) {
    throw std::logic_error("duplicate golden coordinates");
  }
#endif
}

inline DenseToSparseGoldenResult
DenseToSparseGolden(DenseSparseFormat format, int64_t m, int64_t n, int64_t ld,
                    bool rowMajor, int64_t base, size_t valueWidth,
                    bool integer, const std::vector<uint8_t> &dense,
                    int64_t blockSize = 1, int64_t ellCols = 0,
                    const std::vector<int64_t> &ellPattern = {}) {
  const DenseToSparseGoldenArgs args{
      format, m, n, ld, rowMajor, base, valueWidth, integer, dense};
  DenseToSparseValidateGoldenArgs(args);
  if (format == DenseSparseFormat::BLOCKED_ELL) {
    return DenseToSparseBellGolden(args, blockSize, ellCols, ellPattern);
  }
  DenseToSparseGoldenResult result;
  std::vector<std::pair<int64_t, int64_t>> coordinates;
  if (format == DenseSparseFormat::CSC) {
    DenseToSparseScanCsc(args, &coordinates, &result);
  } else {
    DenseToSparseScanRows(args, &coordinates, &result);
  }
  result.nnz = static_cast<int64_t>(coordinates.size());
  DenseToSparseValidateStructure(m, n, coordinates);
  return result;
}

} // namespace sparse_test
#endif
