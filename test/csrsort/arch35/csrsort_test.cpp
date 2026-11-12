/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/**
 * @file csrsort_test.cpp
 * @brief GTest tests for aclsparseXcsrsort (Legacy API, arch35 / ascend950).
 *
 * Covers aclsparseXcsrsort_bufferSizeExt / aclsparseXcsrsort.
 *
 * Sorting semantics - per-row stable sort of csrColInd ascending:
 *   csrColInd is reordered in-place; P uses the same permutation;
 *   csrRowPtr is not modified.
 */

#include "csrsort_golden.h"
#include "sort_tiling_utils.h"
#include "descriptor_manager.h"
#include "fill.h"
#include "sparse_test.h"
#include "tiling/platform/platform_ascendc.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sparse_test;

// ============================================================================
// Global ACL environment
// ============================================================================
class AclTestEnvironment : public testing::Environment {
public:
  void SetUp() override { env_ = std::make_unique<AclEnvScope>(); }
  void TearDown() override { env_.reset(); }
  aclrtStream stream() const { return env_->stream(); }

private:
  std::unique_ptr<AclEnvScope> env_;
};

static AclTestEnvironment *g_acl_env = nullptr;

// ============================================================================
// Test fixture
// ============================================================================
class CsrsortTest : public testing::Test {
public:
  aclsparseHandle_t handle() { return handle_->get(); }
  aclrtStream stream() const { return stream_; }

protected:
  void SetUp() override {
    stream_ = g_acl_env->stream();
    handle_ = std::make_unique<HandleManager>();
    handle_->setStream(stream_);
  }

  void TearDown() override { handle_.reset(); }

  aclrtStream stream_ = nullptr;
  std::unique_ptr<HandleManager> handle_;
};

// ============================================================================
// Helpers
// ============================================================================

::testing::AssertionResult AssertIntVecEq(const std::vector<int32_t> &actual,
                                          const std::vector<int32_t> &expected,
                                          const char *name) {
  if (actual.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << name << " size mismatch: actual=" << actual.size()
           << " expected=" << expected.size();
  }
  for (size_t i = 0; i < actual.size(); i++) {
    if (actual[i] != expected[i]) {
      return ::testing::AssertionFailure()
             << name << "[" << i << "] mismatch: actual=" << actual[i]
             << " expected=" << expected[i];
    }
  }
  return ::testing::AssertionSuccess();
}

static std::vector<int32_t> IdentityPerm(int nnz) {
  std::vector<int32_t> p(nnz);
  std::iota(p.begin(), p.end(), 0);
  return p;
}

static uint64_t GetDeviceUbSize() {
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform == nullptr) {
    return 0U;
  }
  uint64_t ubSize = 0U;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  return ubSize;
}

static ::testing::AssertionResult RunEmptySortAndVerify(
    CsrsortTest *t, int m, int n, const std::vector<int32_t> &rowPtrHost,
    aclsparseIndexBase_t indexBase, const std::string &caseId) {
  size_t bufSize = 42;
  auto ret = aclsparseXcsrsort_bufferSizeExt(
      t->handle(), m, n, 0, reinterpret_cast<const int *>(rowPtrHost.data()),
      nullptr, &bufSize);
  if (ret != ACL_SPARSE_STATUS_SUCCESS) {
    return ::testing::AssertionFailure()
           << caseId << " bufferSizeExt failed: " << ret;
  }
  if (bufSize != 0) {
    return ::testing::AssertionFailure()
           << caseId << " expected bufSize=0 got " << bufSize;
  }
  aclsparseMatDescr_t descr = nullptr;
  aclsparseCreateMatDescr(&descr);
  aclsparseSetMatIndexBase(descr, indexBase);
  ret = aclsparseXcsrsort(t->handle(), m, n, 0, descr,
                          reinterpret_cast<const int *>(rowPtrHost.data()),
                          nullptr, nullptr, nullptr);
  aclsparseDestroyMatDescr(descr);
  if (ret != ACL_SPARSE_STATUS_SUCCESS) {
    return ::testing::AssertionFailure() << caseId << " sort failed: " << ret;
  }
  return ::testing::AssertionSuccess();
}

static ::testing::AssertionResult
VerifySortedOutput(CsrsortTest *t, int m, int nnz, const std::string &caseId,
                   const std::vector<int32_t> &rowPtrHost,
                   const CsrsortGoldenResult &gold, DeviceBuffer &dRowPtr,
                   DeviceBuffer &dColInd, DeviceBuffer &dP) {
  auto syncRet = aclrtSynchronizeStream(t->stream());
  if (syncRet != ACL_SUCCESS) {
    return ::testing::AssertionFailure()
           << caseId << " sync failed: " << syncRet;
  }
  std::vector<int32_t> outRowPtr(m + 1), outCol(nnz), outP(nnz);
  dRowPtr.copyToHost(outRowPtr.data(),
                     static_cast<size_t>(m + 1) * sizeof(int32_t));
  dColInd.copyToHost(outCol.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  dP.copyToHost(outP.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  auto rowPtrResult = AssertIntVecEq(outRowPtr, rowPtrHost, "rowPtr");
  if (!rowPtrResult) {
    return ::testing::AssertionFailure()
           << caseId << " " << rowPtrResult.message();
  }
  auto colResult = AssertIntVecEq(outCol, gold.colInd, "colInd");
  if (!colResult) {
    return ::testing::AssertionFailure()
           << caseId << " " << colResult.message();
  }
  auto pResult = AssertIntVecEq(outP, gold.P, "P");
  if (!pResult) {
    return ::testing::AssertionFailure() << caseId << " " << pResult.message();
  }
  return ::testing::AssertionSuccess();
}

static ::testing::AssertionResult RunNonEmptySortAndVerify(
    CsrsortTest *t, int m, int n, int nnz,
    const std::vector<int32_t> &rowPtrHost,
    const std::vector<int32_t> &colIndHost, aclsparseIndexBase_t indexBase,
    const std::string &caseId, const std::vector<int32_t> *customP) {
  std::vector<int32_t> pHost = customP ? *customP : IdentityPerm(nnz);
  auto gold = csrsortGolden(rowPtrHost, colIndHost, pHost, m,
                            static_cast<int>(indexBase));
  auto dRowPtr = DeviceBuffer::copyFrom(
      rowPtrHost.data(), static_cast<size_t>(m + 1) * sizeof(int32_t));
  auto dColInd = DeviceBuffer::copyFrom(
      colIndHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  auto dP = DeviceBuffer::copyFrom(pHost.data(),
                                   static_cast<size_t>(nnz) * sizeof(int32_t));
  size_t bufSize = 0;
  auto retBuf = aclsparseXcsrsort_bufferSizeExt(
      t->handle(), m, n, nnz, reinterpret_cast<const int *>(dRowPtr.raw()),
      reinterpret_cast<const int *>(dColInd.raw()), &bufSize);
  if (retBuf != ACL_SPARSE_STATUS_SUCCESS) {
    return ::testing::AssertionFailure()
           << caseId << " bufferSizeExt failed: " << retBuf;
  }
  if (bufSize == 0) {
    return ::testing::AssertionFailure()
           << caseId << " bufferSizeExt returned 0 for nnz>0";
  }
  auto dBuf = DeviceBuffer::alloc(bufSize);
  aclsparseMatDescr_t descr = nullptr;
  aclsparseCreateMatDescr(&descr);
  aclsparseSetMatIndexBase(descr, indexBase);
  auto ret = aclsparseXcsrsort(t->handle(), m, n, nnz, descr,
                               reinterpret_cast<const int *>(dRowPtr.raw()),
                               reinterpret_cast<int *>(dColInd.get()),
                               reinterpret_cast<int *>(dP.get()), dBuf.get());
  aclsparseDestroyMatDescr(descr);
  if (ret != ACL_SPARSE_STATUS_SUCCESS) {
    return ::testing::AssertionFailure() << caseId << " sort failed: " << ret;
  }
  return VerifySortedOutput(t, m, nnz, caseId, rowPtrHost, gold, dRowPtr,
                            dColInd, dP);
}

::testing::AssertionResult
RunSortAndVerify(CsrsortTest *t, int m, int n, int nnz,
                 const std::vector<int32_t> &rowPtrHost,
                 const std::vector<int32_t> &colIndHost,
                 aclsparseIndexBase_t indexBase, const std::string &caseId,
                 const std::vector<int32_t> *customP = nullptr) {
  if (nnz == 0) {
    return RunEmptySortAndVerify(t, m, n, rowPtrHost, indexBase, caseId);
  }
  return RunNonEmptySortAndVerify(t, m, n, nnz, rowPtrHost, colIndHost,
                                  indexBase, caseId, customP);
}

// ============================================================================
// Success-path tests
// ============================================================================

// 1. Basic - design doc example
TEST_F(CsrsortTest, Basic) {
  const int m = 3, n = 3, nnz = 9;
  std::vector<int32_t> rowPtr = {0, 3, 6, 9};
  std::vector<int32_t> colInd = {2, 1, 0, 0, 2, 1, 1, 2, 0};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "Basic"));
}

// 2. EmptyRows - first/last/middle empty
TEST_F(CsrsortTest, EmptyRows) {
  const int m = 4, n = 5, nnz = 4;
  std::vector<int32_t> rowPtr = {0, 0, 2, 2, 4};
  std::vector<int32_t> colInd = {3, 1, 4, 2};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "EmptyRows"));
}

// 3. AlreadySorted
TEST_F(CsrsortTest, AlreadySorted) {
  const int m = 3, n = 5, nnz = 6;
  std::vector<int32_t> rowPtr = {0, 2, 4, 6};
  std::vector<int32_t> colInd = {0, 1, 2, 3, 0, 4};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "AlreadySorted"));
}

// 4. Reverse
TEST_F(CsrsortTest, Reverse) {
  const int m = 3, n = 5, nnz = 6;
  std::vector<int32_t> rowPtr = {0, 2, 4, 6};
  std::vector<int32_t> colInd = {4, 0, 3, 2, 4, 1};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "Reverse"));
}

// 5. DuplicateCols - stability
TEST_F(CsrsortTest, DuplicateCols) {
  const int m = 1, n = 5, nnz = 5;
  std::vector<int32_t> rowPtr = {0, 5};
  std::vector<int32_t> colInd = {2, 1, 2, 1, 0};
  std::vector<int32_t> customP = {10, 20, 30, 40, 50};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "DuplicateCols",
                               &customP));
}

// 6. NonIdentityP
TEST_F(CsrsortTest, NonIdentityP) {
  const int m = 2, n = 4, nnz = 4;
  std::vector<int32_t> rowPtr = {0, 2, 4};
  std::vector<int32_t> colInd = {3, 1, 0, 2};
  std::vector<int32_t> customP = {100, 200, 300, 400};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "NonIdentityP",
                               &customP));
}

// 7. Base1
TEST_F(CsrsortTest, Base1) {
  const int m = 2, n = 4, nnz = 4;
  std::vector<int32_t> rowPtr = {1, 3, 5};
  std::vector<int32_t> colInd = {4, 2, 1, 3};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ONE, "Base1"));
}

// 8. EmptyMatrix
TEST_F(CsrsortTest, EmptyMatrix) {
  const int m = 3, n = 3, nnz = 0;
  std::vector<int32_t> rowPtr = {0, 0, 0, 0};
  std::vector<int32_t> colInd;
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "EmptyMatrix"));
}

// 9. SingleElemRows
TEST_F(CsrsortTest, SingleElemRows) {
  const int m = 4, n = 4, nnz = 2;
  std::vector<int32_t> rowPtr = {0, 0, 1, 1, 2};
  std::vector<int32_t> colInd = {3, 1};
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "SingleElemRows"));
}

// 10. ManySmallRows - m > core count
TEST_F(CsrsortTest, ManySmallRows) {
  const int m = 200, n = 10, nnz = 300;
  std::mt19937 rng(99);
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  for (int i = 0; i < m; i++) {
    int len = rng() % 4;
    for (int j = 0; j < len; j++) {
      colInd.push_back(rng() % n);
    }
    rowPtr[i + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(colInd.size()),
                               rowPtr, colInd, ACL_SPARSE_INDEX_BASE_ZERO,
                               "ManySmallRows"));
}

// 11. MultiCoreSkewedRows - irregular row lengths, m exceeds AIV count and
//     several rows force the multi-run path. This covers nnz-weighted row
//     partition boundaries, empty rows and disjoint per-core workspace ranges.
TEST_F(CsrsortTest, MultiCoreSkewedRows) {
  const int m = 257;
  const int n = 4096;
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  for (int row = 0; row < m; row++) {
    int len = 0;
    if (row % 64 == 0) {
      len = 10000;
    } else if (row % 8 == 0) {
      len = 2048;
    } else if (row % 5 != 0) {
      len = row % 7;
    }
    for (int j = 0; j < len; j++) {
      colInd.push_back((row * 131 + len - j) % n);
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(colInd.size()),
                               rowPtr, colInd, ACL_SPARSE_INDEX_BASE_ZERO,
                               "MultiCoreSkewedRows"));
}

// 12. LongRow - single row, multi-run path triggered when len > runSize.
//     Note: multi-run merge-path path uses Sort API per run; currently limited
//     by Sort API reuse on arch35. Uses moderate nnz to exercise single-run.
TEST_F(CsrsortTest, LongRow) {
  const int m = 1, n = 1000, nnz = 1000;
  std::vector<int32_t> rowPtr = {0, nnz};
  std::vector<int32_t> colInd(nnz);
  std::iota(colInd.begin(), colInd.end(), 0);
  std::mt19937 rng(123);
  std::shuffle(colInd.begin(), colInd.end(), rng);
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "LongRow"));
}

// 13. LongRowMultiRun - force multiple UB runs and verify duplicate-key
//     stability across GM merge rounds.
TEST_F(CsrsortTest, LongRowMultiRun) {
  const int m = 1, n = 4096, nnz = 32768;
  std::vector<int32_t> rowPtr = {0, nnz};
  std::vector<int32_t> colInd(nnz);
  for (int i = 0; i < nnz; i++) {
    colInd[i] = i % n;
  }
  std::mt19937 rng(2026);
  std::shuffle(colInd.begin(), colInd.end(), rng);
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "LongRowMultiRun"));
}

// All nonzeros are concentrated in one row. This stresses many bottom-up
// merge rounds, large workspace ping-pong traffic and the final copy-back.
TEST_F(CsrsortTest, SingleHugeRow) {
  constexpr int m = 1;
  constexpr int n = 65536;
  constexpr int nnz = 1024 * 1024;
  const std::vector<int32_t> rowPtr = {0, nnz};
  std::vector<int32_t> colInd(nnz);
  std::vector<int32_t> p(nnz);
  for (int i = 0; i < nnz; i++) {
    // A deterministic permutation with duplicate keys exercises stable sort
    // without the cost and platform variance of shuffling one million items.
    colInd[i] = static_cast<int32_t>(
        (static_cast<uint64_t>(i) * 65521U + 7919U) % 4096U);
    p[i] = i;
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO, "SingleHugeRow",
                               &p));
}

// 14. LargeNnzMultiCoreRows - one million nonzeros distributed over rows
//     that each require multiple UB runs. The repeated keys also verify stable
//     ordering across merge rounds while exercising multiple AIV cores.
TEST_F(CsrsortTest, LargeNnzMultiCoreRows) {
  constexpr int m = 64;
  constexpr int n = 4096;
  constexpr int rowLength = 16384;
  constexpr int nnz = m * rowLength; // 1,048,576
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  colInd.reserve(nnz);
  for (int row = 0; row < m; row++) {
    for (int j = 0; j < rowLength; j++) {
      colInd.push_back((row * 131 + rowLength - j) % n);
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO,
                               "LargeNnzMultiCoreRows"));
}

// 15. RandomMedium
TEST_F(CsrsortTest, RandomMedium) {
  auto csr =
      makeSparseCsr(/*rows=*/50, /*cols=*/50, /*sparsity=*/0.85, /*seed=*/77);
  EXPECT_TRUE(RunSortAndVerify(
      this, static_cast<int>(csr.rows), static_cast<int>(csr.cols),
      static_cast<int>(csr.nnz), csr.rowOffsets, csr.colIndices,
      ACL_SPARSE_INDEX_BASE_ZERO, "RandomMedium"));
}

// 16. NonAlignedBuffer - workspace not 128-byte aligned
TEST_F(CsrsortTest, NonAlignedBuffer) {
  const int m = 2, n = 5, nnz = 4;
  std::vector<int32_t> rowPtr = {0, 2, 4};
  std::vector<int32_t> colInd = {3, 1, 4, 2};

  auto gold = csrsortGolden(rowPtr, colInd, IdentityPerm(nnz), m, 0);

  auto dRowPtr = DeviceBuffer::copyFrom(
      rowPtr.data(), static_cast<size_t>(m + 1) * sizeof(int32_t));
  auto dColInd = DeviceBuffer::copyFrom(
      colInd.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  auto pHost = IdentityPerm(nnz);
  auto dP = DeviceBuffer::copyFrom(pHost.data(),
                                   static_cast<size_t>(nnz) * sizeof(int32_t));

  size_t bufSize = 0;
  ASSERT_EQ(aclsparseXcsrsort_bufferSizeExt(
                handle(), m, n, nnz,
                reinterpret_cast<const int *>(dRowPtr.raw()),
                reinterpret_cast<const int *>(dColInd.raw()), &bufSize),
            ACL_SPARSE_STATUS_SUCCESS);

  // allocate extra and offset by 4 bytes to break 128B alignment
  auto dBufRaw = DeviceBuffer::alloc(bufSize + 128);
  void *dBuf =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(dBufRaw.raw()) + 4);

  aclsparseMatDescr_t descr = nullptr;
  aclsparseCreateMatDescr(&descr);
  aclsparseSetMatIndexBase(descr, ACL_SPARSE_INDEX_BASE_ZERO);
  ASSERT_EQ(aclsparseXcsrsort(handle(), m, n, nnz, descr,
                              reinterpret_cast<const int *>(dRowPtr.raw()),
                              reinterpret_cast<int *>(dColInd.get()),
                              reinterpret_cast<int *>(dP.get()), dBuf),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseDestroyMatDescr(descr);

  ASSERT_EQ(aclrtSynchronizeStream(stream()), ACL_SUCCESS);
  std::vector<int32_t> outRowPtr(m + 1), outCol(nnz), outP(nnz);
  dRowPtr.copyToHost(outRowPtr.data(),
                     static_cast<size_t>(m + 1) * sizeof(int32_t));
  dColInd.copyToHost(outCol.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  dP.copyToHost(outP.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
  EXPECT_TRUE(AssertIntVecEq(outRowPtr, rowPtr, "rowPtr"));
  EXPECT_TRUE(AssertIntVecEq(outCol, gold.colInd, "colInd"));
  EXPECT_TRUE(AssertIntVecEq(outP, gold.P, "P"));
}

// 17. RunSizeBoundaries - exercise the exact UB run boundary on the current
// device.
TEST_F(CsrsortTest, RunSizeBoundaries) {
  uint32_t runSize = 0U;
  uint32_t sortTmpSize = 0U;
  ASSERT_TRUE(
      sort_tiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
  ASSERT_GT(runSize, 1U);
  const int m = 3;
  const int n = 4096;
  const int nnz = static_cast<int>(runSize * 3U);
  std::vector<int32_t> rowPtr = {0, static_cast<int32_t>(runSize - 1U),
                                 static_cast<int32_t>(2U * runSize - 1U), nnz};
  std::vector<int32_t> colInd(nnz);
  for (int i = 0; i < nnz; i++) {
    colInd[i] = (nnz - i) % n;
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, rowPtr, colInd,
                               ACL_SPARSE_INDEX_BASE_ZERO,
                               "RunSizeBoundaries"));
}

// 18. Non64ElementCounts - RADIX_SORT accepts the actual element count; UB
//     buffer capacities are 32-byte aligned independently of calCount.
TEST_F(CsrsortTest, Non64ElementCounts) {
  constexpr int m = 5;
  constexpr int n = 257;
  const std::vector<int32_t> rowLengths = {7, 63, 64, 65, 127};
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  for (int row = 0; row < m; row++) {
    int len = rowLengths[row];
    for (int j = 0; j < len; j++) {
      colInd.push_back((row * 37 + len - j) % n);
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(colInd.size()),
                               rowPtr, colInd, ACL_SPARSE_INDEX_BASE_ZERO,
                               "Non64ElementCounts"));
}

// Exercise repeated TPipe reset/reallocation with sharply changing UB buffer
// sizes. In particular, a large row following a tiny row makes the new MTE2
// destination ranges extend into addresses used by the preceding run.
TEST_F(CsrsortTest, AlternatingTinyAndLargeShortRows) {
  uint32_t runSize = 0U;
  uint32_t sortTmpSize = 0U;
  ASSERT_TRUE(
      sort_tiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
  ASSERT_GT(runSize, 128U);

  constexpr int m = 512;
  constexpr int n = 8192;
  const int largeLen =
      static_cast<int>(std::min<uint32_t>(runSize, 4093U));
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  std::vector<int32_t> p;
  colInd.reserve(static_cast<size_t>(m / 2) * (largeLen + 7));
  p.reserve(colInd.capacity());
  for (int row = 0; row < m; row++) {
    const int len = (row % 2 == 0) ? 7 : largeLen;
    for (int j = 0; j < len; j++) {
      // Many duplicate keys make P validation exercise stable ordering too.
      colInd.push_back((row * 17 + len - j) % 257);
      p.push_back(row * 100000 + j);
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(
      this, m, n, static_cast<int>(colInd.size()), rowPtr, colInd,
      ACL_SPARSE_INDEX_BASE_ZERO, "AlternatingTinyAndLargeShortRows", &p));
}

// Keep every row on the UB Sort path while covering sizes immediately around
// common DMA/vector alignment boundaries.
TEST_F(CsrsortTest, LargeAlignmentBoundaryMatrix) {
  uint32_t runSize = 0U;
  uint32_t sortTmpSize = 0U;
  ASSERT_TRUE(
      sort_tiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
  ASSERT_GT(runSize, 1025U);

  constexpr int m = 640;
  constexpr int n = 4096;
  const int lengths[] = {31,  32,  33,  63,  64,  65,  127,
                         128, 129, 255, 256, 257, 511, 513, 1025};
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  for (int row = 0; row < m; row++) {
    const int len = lengths[row % (sizeof(lengths) / sizeof(lengths[0]))];
    for (int j = 0; j < len; j++) {
      colInd.push_back((row * 193 + j * 61 + len - j) % n);
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(
      this, m, n, static_cast<int>(colInd.size()), rowPtr, colInd,
      ACL_SPARSE_INDEX_BASE_ZERO, "LargeAlignmentBoundaryMatrix"));
}

// Mix empty/single-element rows, maximum-size UB runs and rows just beyond the
// run limit. This repeatedly transitions between the SIMD short-row path and
// the SIMT GM merge path on multiple cores.
TEST_F(CsrsortTest, MixedShortAndMultiRunStress) {
  uint32_t runSize = 0U;
  uint32_t sortTmpSize = 0U;
  ASSERT_TRUE(
      sort_tiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
  ASSERT_GT(runSize, 64U);

  constexpr int m = 193;
  constexpr int n = 16384;
  std::vector<int32_t> rowPtr(m + 1, 0);
  std::vector<int32_t> colInd;
  for (int row = 0; row < m; row++) {
    uint32_t len = 0U;
    switch (row % 8) {
    case 0:
      len = 0U;
      break;
    case 1:
      len = 1U;
      break;
    case 2:
      len = 17U;
      break;
    case 3:
      len = runSize;
      break;
    case 4:
      len = runSize + 1U;
      break;
    case 5:
      len = runSize + 37U;
      break;
    case 6:
      len = 63U;
      break;
    default:
      len = std::min<uint32_t>(runSize, 2047U);
      break;
    }
    for (uint32_t j = 0; j < len; j++) {
      colInd.push_back(static_cast<int32_t>(
          (row * 131U + (len - j) * 29U + j % 11U) % n));
    }
    rowPtr[row + 1] = static_cast<int32_t>(colInd.size());
  }
  EXPECT_TRUE(RunSortAndVerify(
      this, m, n, static_cast<int>(colInd.size()), rowPtr, colInd,
      ACL_SPARSE_INDEX_BASE_ZERO, "MixedShortAndMultiRunStress"));
}

// ============================================================================
// Exception-path tests
// ============================================================================

TEST_F(CsrsortTest, NullHandle) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(nullptr, 4, 4, 1, &dummy, &dummy,
                                            &bufSize),
            ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
  EXPECT_EQ(aclsparseXcsrsort(nullptr, 4, 4, 1, nullptr, &dummy, &dummy, &dummy,
                              &bufSize),
            ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsrsortTest, NullBufferSize) {
  int dummy = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 4, 4, 1, &dummy, &dummy,
                                            nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, InvalidM) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), -1, 4, 0, nullptr,
                                            nullptr, &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, InvalidN) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 4, -1, 0, &dummy, nullptr,
                                            &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, InvalidNnz) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 4, 4, -1, &dummy, &dummy,
                                            &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, EmptyMatrixWithNnz) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 0, 4, 1, nullptr, &dummy,
                                            &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, NullRowPtr) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 4, 4, 1, nullptr, &dummy,
                                            &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, NullColInd) {
  int dummy = 0;
  size_t bufSize = 0;
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(handle(), 4, 4, 1, &dummy, nullptr,
                                            &bufSize),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, NullDescr) {
  int dummy = 0;
  size_t bufSize = 0;
  std::vector<int32_t> rowPtr = {0, 1};
  std::vector<int32_t> colInd = {0};
  auto dRowPtr = DeviceBuffer::copyFrom(rowPtr.data(), 2 * sizeof(int32_t));
  auto dColInd = DeviceBuffer::copyFrom(colInd.data(), sizeof(int32_t));
  EXPECT_EQ(aclsparseXcsrsort_bufferSizeExt(
                handle(), 1, 1, 1, reinterpret_cast<const int *>(dRowPtr.raw()),
                reinterpret_cast<const int *>(dColInd.raw()), &bufSize),
            ACL_SPARSE_STATUS_SUCCESS);
  auto dBuf = DeviceBuffer::alloc(bufSize);
  auto dP = DeviceBuffer::alloc(sizeof(int32_t));
  EXPECT_EQ(aclsparseXcsrsort(handle(), 1, 1, 1, nullptr,
                              reinterpret_cast<const int *>(dRowPtr.raw()),
                              reinterpret_cast<int *>(dColInd.get()),
                              reinterpret_cast<int *>(dP.get()), dBuf.get()),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CsrsortTest, NullP) {
  std::vector<int32_t> rowPtr = {0, 1};
  std::vector<int32_t> colInd = {0};
  auto dRowPtr = DeviceBuffer::copyFrom(rowPtr.data(), 2 * sizeof(int32_t));
  auto dColInd = DeviceBuffer::copyFrom(colInd.data(), sizeof(int32_t));
  size_t bufSize = 0;
  ASSERT_EQ(aclsparseXcsrsort_bufferSizeExt(
                handle(), 1, 1, 1, reinterpret_cast<const int *>(dRowPtr.raw()),
                reinterpret_cast<const int *>(dColInd.raw()), &bufSize),
            ACL_SPARSE_STATUS_SUCCESS);
  auto dBuf = DeviceBuffer::alloc(bufSize);
  aclsparseMatDescr_t descr = nullptr;
  aclsparseCreateMatDescr(&descr);
  EXPECT_EQ(aclsparseXcsrsort(handle(), 1, 1, 1, descr,
                              reinterpret_cast<const int *>(dRowPtr.raw()),
                              reinterpret_cast<int *>(dColInd.get()), nullptr,
                              dBuf.get()),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  aclsparseDestroyMatDescr(descr);
}

TEST_F(CsrsortTest, NullBuffer) {
  std::vector<int32_t> rowPtr = {0, 1};
  std::vector<int32_t> colInd = {0};
  auto dRowPtr = DeviceBuffer::copyFrom(rowPtr.data(), 2 * sizeof(int32_t));
  auto dColInd = DeviceBuffer::copyFrom(colInd.data(), sizeof(int32_t));
  auto dP = DeviceBuffer::alloc(sizeof(int32_t));
  aclsparseMatDescr_t descr = nullptr;
  aclsparseCreateMatDescr(&descr);
  EXPECT_EQ(aclsparseXcsrsort(handle(), 1, 1, 1, descr,
                              reinterpret_cast<const int *>(dRowPtr.raw()),
                              reinterpret_cast<int *>(dColInd.get()),
                              reinterpret_cast<int *>(dP.get()), nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  aclsparseDestroyMatDescr(descr);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  g_acl_env = new AclTestEnvironment();
  testing::AddGlobalTestEnvironment(g_acl_env);
  return RUN_ALL_TESTS();
}
