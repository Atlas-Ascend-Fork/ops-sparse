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

#ifndef TEST_DENSETOSPARSE_ARCH35_NPU_WRAPPER_H_
#define TEST_DENSETOSPARSE_ARCH35_NPU_WRAPPER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "densetosparse_golden.h"
#include "densetosparse_param.h"
#include "descriptor_manager.h"

namespace sparse_test {

inline size_t DenseToSparseTypeSize(aclDataType type) {
  if (type == ACL_INT8)
    return 1;
  if (type == ACL_FLOAT16 || type == ACL_BF16)
    return 2;
  if (type == ACL_FLOAT)
    return 4;
  if (type == ACL_COMPLEX64)
    return 8;
  return 0;
}

inline aclDataType DenseToSparseValueType(const std::string &name) {
  if (name == "INT8")
    return ACL_INT8;
  if (name == "FP16")
    return ACL_FLOAT16;
  if (name == "BF16")
    return ACL_BF16;
  if (name == "FP32")
    return ACL_FLOAT;
  if (name == "CPLX64")
    return ACL_COMPLEX64;
  if (name == "FP64")
    return ACL_DOUBLE;
  return static_cast<aclDataType>(-1);
}

inline aclsparseIndexType_t DenseToSparseIndexType(const std::string &name) {
  return name == "I64" ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
}

inline aclsparseIndexBase_t DenseToSparseBase(const std::string &name) {
  return name == "ONE" ? ACL_SPARSE_INDEX_BASE_ONE : ACL_SPARSE_INDEX_BASE_ZERO;
}

inline aclsparseOrder_t DenseToSparseOrder(const std::string &name) {
  return name == "COL" ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
}

inline DenseSparseFormat DenseToSparseFormatOf(const std::string &name) {
  if (name == "CSR")
    return DenseSparseFormat::CSR;
  if (name == "CSC")
    return DenseSparseFormat::CSC;
  if (name == "COO")
    return DenseSparseFormat::COO;
  return DenseSparseFormat::BLOCKED_ELL;
}

inline void StoreDenseBits(std::vector<uint8_t> *dense, size_t element,
                           size_t width, uint64_t bits) {
  for (size_t byte = 0; byte < width; ++byte) {
    (*dense)[element * width + byte] =
        static_cast<uint8_t>((bits >> (byte * 8)) & 0xffu);
  }
}

struct DenseToSparseHostInput {
  std::vector<uint8_t> dense;
  std::vector<int64_t> ellPattern;
};

inline bool DenseToSparseIsNonzero(const DenseToSparseParam &p, uint64_t linear,
                                   uint64_t logical) {
  if (p.distribution == "Z100" || p.distribution == "BELL_EMPTY")
    return false;
  if (p.distribution == "Z99" || p.distribution == "SINGLE") {
    return linear == p.seed % std::max<uint64_t>(logical, 1);
  }
  if (p.distribution == "Z75" || p.distribution == "BELL_Z75") {
    return linear % 4 == 0;
  }
  if (p.distribution == "Z50" || p.distribution == "BELL_Z50") {
    return linear % 2 == 0;
  }
  return p.distribution != "Z25" || linear % 4 != 0;
}

inline uint64_t DenseToSparseInputBits(aclDataType type, uint64_t linear,
                                       uint64_t seed) {
  if (type == ACL_INT8) {
    uint64_t bits =
        static_cast<uint8_t>(static_cast<int8_t>((linear + seed) % 255 - 127));
    return bits == 0 ? 1 : bits;
  }
  if (type == ACL_FLOAT)
    return 0x3f000001u + static_cast<uint32_t>(linear % 127);
  if (type == ACL_COMPLEX64) {
    // Alternate which component carries the magnitude so that cases with a
    // zero real part or a zero imaginary part still count as non-zero.
    const uint64_t re = linear % 3 == 0 ? 0u : 0x3f000001u + (linear % 127);
    const uint64_t im = linear % 3 == 1 ? 0u : 0xbf000001u + (linear % 113);
    return (im << 32) | re;
  }
  if (type == ACL_FLOAT16)
    return 0x3801u + static_cast<uint16_t>(linear % 63);
  return 0x3f01u + static_cast<uint16_t>(linear % 63);
}

inline void FillDenseToSparseValues(DenseToSparseHostInput *input,
                                    const DenseToSparseParam &p,
                                    aclDataType type, size_t width) {
  const uint64_t logical = static_cast<uint64_t>(p.m) * p.n;
  for (int64_t row = 0; row < p.m; ++row) {
    for (int64_t col = 0; col < p.n; ++col) {
      const uint64_t linear = static_cast<uint64_t>(row) * p.n + col;
      const size_t element =
          DenseToSparsePhysicalIndex(row, col, p.ld, p.order == "ROW");
      const uint32_t bits = DenseToSparseIsNonzero(p, linear, logical)
                                ? DenseToSparseInputBits(type, linear, p.seed)
                                : 0;
      StoreDenseBits(&input->dense, element, width, bits);
    }
  }
}

// Signed zeros first (both excluded from the structure), then NaN, infinities
// and an ordinary value that must all be preserved bit for bit.
inline uint64_t DenseToSparseSpecialBits(aclDataType type, size_t k) {
  const uint32_t special32[] = {0x00000000u, 0x80000000u, 0x7fc12345u,
                                0x7f800000u, 0xff800000u, 0x3fc00000u};
  // complex64 is packed as (imaginary << 32) | real, so these cover a zero
  // pair with mixed signs plus a non-zero component in either half.
  const uint64_t complex64[] = {0x0000000000000000ull, 0x8000000000000000ull,
                                0x000000007fc12345ull, 0x7fc1234500000000ull,
                                0xff8000007f800000ull, 0xbfc000003fc00000ull};
  const uint16_t special16[] = {
      0x0000u,
      0x8000u,
      static_cast<uint16_t>(type == ACL_FLOAT16 ? 0x7e55u : 0x7fc5u),
      static_cast<uint16_t>(type == ACL_FLOAT16 ? 0x7c00u : 0x7f80u),
      static_cast<uint16_t>(type == ACL_FLOAT16 ? 0xfc00u : 0xff80u),
      0x3f00u};
  if (type == ACL_COMPLEX64)
    return complex64[k];
  if (type == ACL_FLOAT)
    return special32[k];
  return special16[k];
}

inline void FillDenseToSparseSpecial(DenseToSparseHostInput *input,
                                     const DenseToSparseParam &p,
                                     aclDataType type, size_t width) {
  const uint64_t logical = static_cast<uint64_t>(p.m) * p.n;
  if (p.distribution == "SPECIAL" && logical >= 6 && type != ACL_INT8) {
    for (size_t k = 0; k < 6; ++k) {
      const int64_t row = static_cast<int64_t>(k) / p.n;
      const int64_t col = static_cast<int64_t>(k) % p.n;
      const size_t element =
          DenseToSparsePhysicalIndex(row, col, p.ld, p.order == "ROW");
      StoreDenseBits(&input->dense, element, width,
                     DenseToSparseSpecialBits(type, k));
    }
  }
}

inline void FillDenseToSparseBellPattern(DenseToSparseHostInput *input,
                                         const DenseToSparseParam &p) {
  if (p.format != "BELL")
    return;
  const int64_t slots = DenseToSparseCeilDiv(p.ell_cols, p.block_size);
  const int64_t blockRows = DenseToSparseCeilDiv(p.m, p.block_size);
  const int64_t blockColumns = DenseToSparseCeilDiv(p.n, p.block_size);
  input->ellPattern.resize(static_cast<size_t>(blockRows * slots), -1);
  // 退化输入下块列数可能为 0，此时没有任何块列可供指向，pattern 保持全 -1。
  // DenseToSparseCeilDiv 不会返回负数，所以这里判 0 即可覆盖全部退化情形。
  if (blockColumns == 0) {
    return;
  }
  const int64_t base = p.base == "ONE" ? 1 : 0;
  for (int64_t blockRow = 0; blockRow < blockRows; ++blockRow) {
    for (int64_t slot = 0; slot < slots; ++slot) {
      const bool padding =
          p.distribution.find("PAD") != std::string::npos && slot == slots - 1;
      const bool sparsePadding =
          (p.distribution == "BELL_Z50" && slot % 2 != 0) ||
          (p.distribution == "BELL_Z75" && slot % 4 != 0);
      if (!padding && !sparsePadding) {
        input->ellPattern[static_cast<size_t>(blockRow * slots + slot)] =
            (slot + blockRow) % blockColumns + base;
      }
    }
  }
  if (p.distribution == "BELL_OOB" && !input->ellPattern.empty()) {
    input->ellPattern.front() = -2;
    if (input->ellPattern.size() > 1) {
      input->ellPattern.back() = blockColumns + base;
    }
  }
}

inline DenseToSparseHostInput
MakeDenseToSparseInput(const DenseToSparseParam &p) {
  const aclDataType type = DenseToSparseValueType(p.value_type);
  const size_t width = DenseToSparseTypeSize(type);
  if (width == 0 || p.m < 0 || p.n < 0 || p.ld < 0) {
    throw std::invalid_argument("non-executable DenseToSparse parameter");
  }
  const size_t physical =
      static_cast<size_t>((p.order == "ROW" ? p.m : p.n) * p.ld);
  DenseToSparseHostInput input;
  input.dense.assign(physical * width, 0);
  FillDenseToSparseValues(&input, p, type, width);
  FillDenseToSparseSpecial(&input, p, type, width);
  FillDenseToSparseBellPattern(&input, p);
  return input;
}

inline std::vector<uint8_t>
PackDenseToSparseIndices(const std::vector<int64_t> &values,
                         aclsparseIndexType_t type) {
  const size_t width = type == ACL_SPARSE_INDEX_64I ? 8 : 4;
  std::vector<uint8_t> bytes(values.size() * width);
  for (size_t i = 0; i < values.size(); ++i) {
    if (width == 8) {
      const int64_t value = values[i];
      const auto *begin = reinterpret_cast<const uint8_t *>(&value);
      std::copy_n(begin, width, bytes.begin() + i * width);
    } else {
      const int32_t value = static_cast<int32_t>(values[i]);
      const auto *begin = reinterpret_cast<const uint8_t *>(&value);
      std::copy_n(begin, width, bytes.begin() + i * width);
    }
  }
  return bytes;
}

inline std::unique_ptr<DeviceBuffer>
DenseToSparseCopy(const std::vector<uint8_t> &bytes) {
  if (bytes.empty())
    return nullptr;
  return std::make_unique<DeviceBuffer>(
      DeviceBuffer::copyFrom(bytes.data(), bytes.size()));
}

class DenseToSparseSpMat {
public:
  ~DenseToSparseSpMat() {
    if (descr_ != nullptr)
      aclsparseDestroySpMat(descr_);
  }
  aclsparseSpMatDescr_t get() const { return descr_; }
  aclsparseStatus_t create(const DenseToSparseParam &p, void *ptrs,
                           void *index0, void *index1, void *values,
                           void *ellPattern) {
    const auto ot = DenseToSparseIndexType(p.offset_type);
    const auto it = DenseToSparseIndexType(p.index_type);
    const auto base = DenseToSparseBase(p.base);
    const auto vt = DenseToSparseValueType(p.value_type);
    if (p.format == "CSR") {
      return aclsparseCreateCsr(&descr_, p.m, p.n, 0, ptrs, index0, values, ot,
                                it, base, vt);
    }
    if (p.format == "CSC") {
      return aclsparseCreateCsc(&descr_, p.m, p.n, 0, ptrs, index0, values, ot,
                                it, base, vt);
    }
    if (p.format == "COO") {
      return aclsparseCreateCoo(&descr_, p.m, p.n, 0, index0, index1, values,
                                it, base, vt);
    }
    return aclsparseCreateBlockedEll(&descr_, p.m, p.n, p.block_size,
                                     p.ell_cols, ellPattern, values, it, base,
                                     vt);
  }

private:
  aclsparseSpMatDescr_t descr_ = nullptr;
};

struct DenseToSparseRunResult {
  aclsparseStatus_t descriptorStatus = ACL_SPARSE_STATUS_SUCCESS;
  aclsparseStatus_t queryStatus = ACL_SPARSE_STATUS_SUCCESS;
  aclsparseStatus_t analysisStatus = ACL_SPARSE_STATUS_SUCCESS;
  aclsparseStatus_t setPointersStatus = ACL_SPARSE_STATUS_SUCCESS;
  aclsparseStatus_t convertStatus = ACL_SPARSE_STATUS_SUCCESS;
  aclError syncStatus = ACL_SUCCESS;
  size_t workspaceSize = 0;
  uintptr_t analysisWorkspaceAddress = 0;
  uintptr_t convertWorkspaceAddress = 0;
  int64_t queriedNnz = 0;
  int64_t bellNnzAfterQuery = -1;
  int64_t bellNnzAfterAnalysis = -1;
  int64_t bellNnzAfterConvert = -1;
  std::vector<uint8_t> offsets;
  std::vector<uint8_t> indices0;
  std::vector<uint8_t> indices1;
  std::vector<uint8_t> values;
  std::vector<uint8_t> ellPattern;
};

} // namespace sparse_test

#include "densetosparse_npu_execution.h"
#endif
