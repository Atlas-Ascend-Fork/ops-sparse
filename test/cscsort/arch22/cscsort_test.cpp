/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/**
 * @file cscsort_test.cpp
 * @brief GTest tests for aclsparseXcscsort (Legacy API, arch22 / Ascend 910B/910C).
 *
 * Column-wise dual of aclsparseXcsrsort. Covers
 * aclsparseXcscsort_bufferSizeExt / aclsparseXcscsort.
 *
 * Sorting semantics - per-column stable sort of cscRowInd ascending:
 *   cscRowInd is reordered in-place; P uses the same permutation;
 *   cscColPtr is not modified.
 */

#include "sparse_test.h"
#include "fill.h"
#include "descriptor_manager.h"
#include "cscsort_golden.h"
#include "cscsort_tiling_data.h"
#include "cscsort_tiling_utils.h"
#include "../../frame/presmoke.h"
#include "tiling/platform/platform_ascendc.h"

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
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
    void SetUp() override
    {
        env_ = std::make_unique<AclEnvScope>();
    }
    void TearDown() override
    {
        env_.reset();
    }
    aclrtStream stream() const
    {
        return env_->stream();
    }

private:
    std::unique_ptr<AclEnvScope> env_;
};

static AclTestEnvironment *g_acl_env = nullptr;

// ============================================================================
// Test fixture
// ============================================================================
class CscsortTest : public testing::Test {
public:
    aclsparseHandle_t handle()
    {
        return handle_->get();
    }
    aclrtStream stream() const
    {
        return stream_;
    }

protected:
    void SetUp() override
    {
        stream_ = g_acl_env->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);
    }

    void TearDown() override
    {
        handle_.reset();
    }

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
};

// ============================================================================
// Helpers
// ============================================================================

::testing::AssertionResult AssertIntVecEq(
    const std::vector<int32_t> &actual, const std::vector<int32_t> &expected, const char *name)
{
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << name << " size mismatch: actual=" << actual.size() << " expected=" << expected.size();
    }
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            return ::testing::AssertionFailure()
                   << name << "[" << i << "] mismatch: actual=" << actual[i] << " expected=" << expected[i];
        }
    }
    return ::testing::AssertionSuccess();
}

static std::vector<int32_t> IdentityPerm(int nnz)
{
    std::vector<int32_t> p(nnz);
    std::iota(p.begin(), p.end(), 0);
    return p;
}

static uint64_t GetDeviceUbSize()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return 0U;
    }
    uint64_t ubSize = 0U;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return ubSize;
}

static uint32_t GetDeviceAivCoreCount()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return 0U;
    }
    return platform->GetCoreNumAiv();
}

static ::testing::AssertionResult RunEmptySortAndVerify(CscsortTest *t, int m, int n,
    const std::vector<int32_t> &colPtrHost, aclsparseIndexBase_t indexBase, const std::string &caseId)
{
    size_t bufSize = 42;
    auto ret = aclsparseXcscsort_bufferSizeExt(t->handle(), m, n, 0,
        reinterpret_cast<const int *>(colPtrHost.data()), nullptr, &bufSize);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " bufferSizeExt failed: " << ret;
    }
    if (bufSize != 0) {
        return ::testing::AssertionFailure() << caseId << " expected bufSize=0 got " << bufSize;
    }
    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    aclsparseSetMatIndexBase(descr, indexBase);
    ret = aclsparseXcscsort(t->handle(), m, n, 0, descr,
        reinterpret_cast<const int *>(colPtrHost.data()), nullptr, nullptr, nullptr);
    aclsparseDestroyMatDescr(descr);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " sort failed: " << ret;
    }
    return ::testing::AssertionSuccess();
}

static ::testing::AssertionResult VerifySortedOutput(CscsortTest *t, int n, int nnz,
    const std::string &caseId, const std::vector<int32_t> &colPtrHost,
    const CscsortGoldenResult &gold, DeviceBuffer &dColPtr, DeviceBuffer &dRowInd, DeviceBuffer &dP)
{
    auto syncRet = aclrtSynchronizeStream(t->stream());
    if (syncRet != ACL_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " sync failed: " << syncRet;
    }
    std::vector<int32_t> outColPtr(n + 1), outRow(nnz), outP(nnz);
    dColPtr.copyToHost(outColPtr.data(), static_cast<size_t>(n + 1) * sizeof(int32_t));
    dRowInd.copyToHost(outRow.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    dP.copyToHost(outP.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto colPtrResult = AssertIntVecEq(outColPtr, colPtrHost, "colPtr");
    if (!colPtrResult) {
        return ::testing::AssertionFailure() << caseId << " " << colPtrResult.message();
    }
    auto rowResult = AssertIntVecEq(outRow, gold.rowInd, "rowInd");
    if (!rowResult) {
        return ::testing::AssertionFailure() << caseId << " " << rowResult.message();
    }
    auto pResult = AssertIntVecEq(outP, gold.P, "P");
    if (!pResult) {
        return ::testing::AssertionFailure() << caseId << " " << pResult.message();
    }
    return ::testing::AssertionSuccess();
}

static ::testing::AssertionResult RunNonEmptySortAndVerify(CscsortTest *t, int m, int n, int nnz,
    const std::vector<int32_t> &colPtrHost, const std::vector<int32_t> &rowIndHost,
    aclsparseIndexBase_t indexBase, const std::string &caseId, const std::vector<int32_t> *customP)
{
    std::vector<int32_t> pHost = customP ? *customP : IdentityPerm(nnz);
    auto gold = cscsortGolden(colPtrHost, rowIndHost, pHost, n, static_cast<int>(indexBase));
    auto dColPtr = DeviceBuffer::copyFrom(colPtrHost.data(), static_cast<size_t>(n + 1) * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowIndHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto dP = DeviceBuffer::copyFrom(pHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    size_t bufSize = 0;
    auto retBuf = aclsparseXcscsort_bufferSizeExt(t->handle(), m, n, nnz,
        reinterpret_cast<const int *>(dColPtr.raw()), reinterpret_cast<const int *>(dRowInd.raw()), &bufSize);
    if (retBuf != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " bufferSizeExt failed: " << retBuf;
    }
    if (bufSize == 0) {
        return ::testing::AssertionFailure() << caseId << " bufferSizeExt returned 0 for nnz>0";
    }
    auto dBuf = DeviceBuffer::alloc(bufSize);
    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    aclsparseSetMatIndexBase(descr, indexBase);
    auto ret = aclsparseXcscsort(t->handle(), m, n, nnz, descr,
        reinterpret_cast<const int *>(dColPtr.raw()), reinterpret_cast<int *>(dRowInd.get()),
        reinterpret_cast<int *>(dP.get()), dBuf.get());
    aclsparseDestroyMatDescr(descr);
    if (ret != ACL_SPARSE_STATUS_SUCCESS) {
        return ::testing::AssertionFailure() << caseId << " sort failed: " << ret;
    }
    return VerifySortedOutput(t, n, nnz, caseId, colPtrHost, gold, dColPtr, dRowInd, dP);
}

::testing::AssertionResult RunSortAndVerify(CscsortTest *t, int m, int n, int nnz,
    const std::vector<int32_t> &colPtrHost, const std::vector<int32_t> &rowIndHost,
    aclsparseIndexBase_t indexBase, const std::string &caseId, const std::vector<int32_t> *customP = nullptr)
{
    if (nnz == 0) {
        return RunEmptySortAndVerify(t, m, n, colPtrHost, indexBase, caseId);
    }
    return RunNonEmptySortAndVerify(t, m, n, nnz, colPtrHost, rowIndHost, indexBase, caseId, customP);
}

// ============================================================================
// L0 - threshold tests
// ============================================================================

// L0-1 Basic - design doc example, each column shuffled
TEST_F(CscsortTest, Basic)
{
    const int m = 3, n = 3, nnz = 9;
    std::vector<int32_t> colPtr = {0, 3, 6, 9};
    std::vector<int32_t> rowInd = {2, 1, 0, 0, 2, 1, 1, 2, 0};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "Basic"));
}

// L0-2 EmptyMatrix - nnz == 0
TEST_F(CscsortTest, EmptyMatrix)
{
    const int m = 3, n = 3, nnz = 0;
    std::vector<int32_t> colPtr = {0, 0, 0, 0};
    std::vector<int32_t> rowInd;
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "EmptyMatrix"));
}

// ============================================================================
// L1 - functional tests
// ============================================================================

// L1-1 AlreadySorted - each column already ascending
TEST_F(CscsortTest, AlreadySorted)
{
    const int m = 5, n = 3, nnz = 6;
    std::vector<int32_t> colPtr = {0, 2, 4, 6};
    std::vector<int32_t> rowInd = {0, 1, 2, 3, 0, 4};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "AlreadySorted"));
}

// L1-2 Reverse - each column descending
TEST_F(CscsortTest, Reverse)
{
    const int m = 5, n = 3, nnz = 6;
    std::vector<int32_t> colPtr = {0, 2, 4, 6};
    std::vector<int32_t> rowInd = {4, 0, 3, 2, 4, 1};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "Reverse"));
}

// L1-3 DuplicateRows - stability with duplicate rowInd and custom P
TEST_F(CscsortTest, DuplicateRows)
{
    const int m = 5, n = 1, nnz = 5;
    std::vector<int32_t> colPtr = {0, 5};
    std::vector<int32_t> rowInd = {2, 1, 2, 1, 0};
    std::vector<int32_t> customP = {10, 20, 30, 40, 50};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO,
                                 "DuplicateRows", &customP));
}

// L1-4 NonIdentityP - custom permutation values
TEST_F(CscsortTest, NonIdentityP)
{
    const int m = 4, n = 2, nnz = 4;
    std::vector<int32_t> colPtr = {0, 2, 4};
    std::vector<int32_t> rowInd = {3, 1, 0, 2};
    std::vector<int32_t> customP = {100, 200, 300, 400};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO,
                                 "NonIdentityP", &customP));
}

// L1-5 Base1 - indexBase == 1, colPtr starts from 1
TEST_F(CscsortTest, Base1)
{
    const int m = 4, n = 2, nnz = 4;
    std::vector<int32_t> colPtr = {1, 3, 5};
    std::vector<int32_t> rowInd = {4, 2, 1, 3};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ONE, "Base1"));
}

// L1-6 EmptyCols - first/middle empty columns
TEST_F(CscsortTest, EmptyCols)
{
    const int m = 5, n = 4, nnz = 4;
    std::vector<int32_t> colPtr = {0, 0, 2, 2, 4};
    std::vector<int32_t> rowInd = {3, 1, 4, 2};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "EmptyCols"));
}

// L1-7 SingleElemCols - some columns have only one element
TEST_F(CscsortTest, SingleElemCols)
{
    const int m = 4, n = 4, nnz = 2;
    std::vector<int32_t> colPtr = {0, 0, 1, 1, 2};
    std::vector<int32_t> rowInd = {3, 1};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "SingleElemCols"));
}

// L1-8 ManySmallCols - n=200 > AIV core count, each column 0-3 elements
TEST_F(CscsortTest, ManySmallCols)
{
    const int m = 10, n = 200;
    std::mt19937 rng(99);
    std::vector<int32_t> colPtr(n + 1, 0);
    std::vector<int32_t> rowInd;
    for (int j = 0; j < n; j++) {
        int len = rng() % 4;
        for (int k = 0; k < len; k++) {
            rowInd.push_back(rng() % m);
        }
        colPtr[j + 1] = static_cast<int32_t>(rowInd.size());
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(rowInd.size()), colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "ManySmallCols"));
}

// L1-9 MultiCoreSkewedCols - irregular column lengths, n exceeds AIV count and
//     several columns force the multi-run path. Covers nnz-weighted column
//     partition boundaries, empty columns and disjoint per-core workspace.
TEST_F(CscsortTest, MultiCoreSkewedCols)
{
    const int m = 4096;
    const int n = 257;
    std::vector<int32_t> colPtr(n + 1, 0);
    std::vector<int32_t> rowInd;
    for (int col = 0; col < n; col++) {
        int len = 0;
        if (col % 64 == 0) {
            len = 10000;
        } else if (col % 8 == 0) {
            len = 2048;
        } else if (col % 5 != 0) {
            len = col % 7;
        }
        for (int j = 0; j < len; j++) {
            rowInd.push_back((col * 131 + len - j) % m);
        }
        colPtr[col + 1] = static_cast<int32_t>(rowInd.size());
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(rowInd.size()), colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "MultiCoreSkewedCols"));
}

// L1-10 LongCol - single shuffled column. On devices where runSize >= nnz
//     (e.g. A2 UB-derived runSize ~3600 > 1000) this stays on the short-column
//     single-run path; LongColMultiRun below forces the actual multi-run
//     GM merge.
TEST_F(CscsortTest, LongCol)
{
    const int m = 1000, n = 1, nnz = 1000;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    std::iota(rowInd.begin(), rowInd.end(), 0);
    std::mt19937 rng(123);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "LongCol"));
}

// L1-11 LongColMultiRun - force multiple UB runs and verify duplicate-key
//     stability across GM merge rounds.
TEST_F(CscsortTest, LongColMultiRun)
{
    const int m = 4096, n = 1, nnz = 32768;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::mt19937 rng(2026);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(
        this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "LongColMultiRun"));
}

// L1-12 RandomMedium - random CSC matrix from frame generator
TEST_F(CscsortTest, RandomMedium)
{
    auto csc = makeSparseCsc(/*rows=*/50, /*cols=*/50, /*sparsity=*/0.85, /*seed=*/77);
    EXPECT_TRUE(RunSortAndVerify(this,
        static_cast<int>(csc.rows), static_cast<int>(csc.cols), static_cast<int>(csc.nnz),
        csc.colOffsets, csc.rowIndices, ACL_SPARSE_INDEX_BASE_ZERO, "RandomMedium"));
}

// ============================================================================
// L2 - boundary and exception tests
// ============================================================================

// L2-1 RunSizeBoundaries - exercise the exact UB run boundary on the current
//     device. Three columns with length runSize-1 / runSize / runSize+1.
TEST_F(CscsortTest, RunSizeBoundaries)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    const int m = 4096;
    const int n = 3;
    const int nnz = static_cast<int>(runSize * 3U);
    std::vector<int32_t> colPtr = {
        0, static_cast<int32_t>(runSize - 1U), static_cast<int32_t>(2U * runSize - 1U), nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = (nnz - i) % m;
    }
    EXPECT_TRUE(RunSortAndVerify(
        this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "RunSizeBoundaries"));
}

// L2-2 NonAlignedBuffer - workspace not 128-byte aligned.
//     Per requirement §2.8, pBuffer address must be 128-byte aligned, otherwise
//     the operator returns ACL_SPARSE_STATUS_INVALID_VALUE.
TEST_F(CscsortTest, NonAlignedBuffer)
{
    const int m = 5, n = 2, nnz = 4;
    std::vector<int32_t> colPtr = {0, 2, 4};
    std::vector<int32_t> rowInd = {3, 1, 4, 2};

    auto dColPtr = DeviceBuffer::copyFrom(colPtr.data(), static_cast<size_t>(n + 1) * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowInd.data(), static_cast<size_t>(nnz) * sizeof(int32_t));
    auto pHost = IdentityPerm(nnz);
    auto dP = DeviceBuffer::copyFrom(pHost.data(), static_cast<size_t>(nnz) * sizeof(int32_t));

    size_t bufSize = 0;
    ASSERT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), m, n, nnz,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<const int *>(dRowInd.raw()), &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);

    // allocate extra and offset by 4 bytes to break 128B alignment
    auto dBufRaw = DeviceBuffer::alloc(bufSize + 128);
    void *dBuf = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(dBufRaw.raw()) + 4);

    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    aclsparseSetMatIndexBase(descr, ACL_SPARSE_INDEX_BASE_ZERO);
    EXPECT_EQ(aclsparseXcscsort(handle(), m, n, nnz, descr,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<int *>(dRowInd.get()),
                  reinterpret_cast<int *>(dP.get()), dBuf),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyMatDescr(descr);
}

// L2-3 LargeNnzMultiCore - one million nonzeros distributed over columns
//     that each require multiple UB runs. Repeated keys verify stable ordering
//     across merge rounds while exercising multiple AIV cores.
TEST_F(CscsortTest, LargeNnzMultiCore)
{
    constexpr int m = 4096;
    constexpr int n = 64;
    constexpr int colLength = 16384;
    constexpr int nnz = n * colLength;  // 1,048,576
    std::vector<int32_t> colPtr(n + 1, 0);
    std::vector<int32_t> rowInd;
    rowInd.reserve(nnz);
    for (int col = 0; col < n; col++) {
        for (int j = 0; j < colLength; j++) {
            rowInd.push_back((col * 131 + colLength - j) % m);
        }
        colPtr[col + 1] = static_cast<int32_t>(rowInd.size());
    }
    EXPECT_TRUE(RunSortAndVerify(
        this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "LargeNnzMultiCore"));
}

// ============================================================================
// CP2.1 supplementary tests (M-1 / M-2)
// ============================================================================

// M-1 LastColEmpty - last column has no elements: colPtr={0,2,4,4}, col 2 empty
TEST_F(CscsortTest, LastColEmpty)
{
    const int m = 5, n = 3, nnz = 4;
    std::vector<int32_t> colPtr = {0, 2, 4, 4};
    std::vector<int32_t> rowInd = {3, 1, 4, 2};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "LastColEmpty"));
}

// M-2 SingleRow - m=1 single-row matrix, every non-empty column is a
//     single-element column (rowInd in [0,1)).
TEST_F(CscsortTest, SingleRow)
{
    const int m = 1, n = 3, nnz = 3;
    std::vector<int32_t> colPtr = {0, 1, 2, 3};
    std::vector<int32_t> rowInd = {0, 0, 0};
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd, ACL_SPARSE_INDEX_BASE_ZERO, "SingleRow"));
}

// ============================================================================
// White-box tests (phase 3.3)
//
// Source-driven branch coverage for kernel and host paths not deterministically
// exercised by the black-box suite. Each test targets a specific execution
// branch identified in the source analysis.
// ============================================================================

// Wb-1 InvalidIndexBase - host validation branch (cscsort_host.cpp:190-193).
//     aclsparseSetMatIndexBase does not validate the enum value, so we can
//     inject an invalid indexBase (2) and verify the host rejects it.
TEST_F(CscsortTest, WbInvalidIndexBase)
{
    std::vector<int32_t> colPtr = {0, 2};
    std::vector<int32_t> rowInd = {1, 0};
    auto dColPtr = DeviceBuffer::copyFrom(colPtr.data(), 2 * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowInd.data(), 2 * sizeof(int32_t));
    size_t bufSize = 0;
    ASSERT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 2, 1, 2,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<const int *>(dRowInd.raw()), &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    auto dBuf = DeviceBuffer::alloc(bufSize);
    auto dP = DeviceBuffer::alloc(2 * sizeof(int32_t));
    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    aclsparseSetMatIndexBase(descr, static_cast<aclsparseIndexBase_t>(2));
    EXPECT_EQ(aclsparseXcscsort(handle(), 2, 1, 2, descr,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<int *>(dRowInd.get()),
                  reinterpret_cast<int *>(dP.get()), dBuf.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyMatDescr(descr);
}

// Wb-2 MergeTwoRunsNoCopyBack - deterministic runCount=2.
//     len = runSize + 1 → runCount = 2 → one merge round (scratch→orig),
//     lN_ becomes 1, break at line 176. resultInScratch=false → no CopyBack.
//     Covers: SortMultiRun break path, no-CopyBack path.
TEST_F(CscsortTest, WbMergeTwoRunsNoCopyBack)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    const int nnz = static_cast<int>(runSize + 1U);
    const int m = nnz / 2 + 1;
    const int n = 1;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::mt19937 rng(42);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbMergeTwoRunsNoCopyBack"));
}

// Wb-3 MergeThreeRunsCopyBack - deterministic runCount=3.
//     len = 2*runSize + 1 → runCount = 3.
//     Round 1 (scratch→orig): pairs (0,1), (2,empty). Odd pair has bLen=0.
//     Round 2 (orig→scratch): pair (0,1). lN_=1, loop ends.
//     resultInScratch=true → CopyBack executed.
//     Covers: CopyBack path, odd pair bLen=0, MergeSegment bCount=0.
TEST_F(CscsortTest, WbMergeThreeRunsCopyBack)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    const int nnz = static_cast<int>(2U * runSize + 1U);
    const int m = nnz / 3 + 1;
    const int n = 1;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::mt19937 rng(43);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbMergeThreeRunsCopyBack"));
}

// Wb-4 MergeFiveRunsNoCopyBack - deterministic runCount=5, three merge rounds.
//     len = 4*runSize + 1 → runCount = 5.
//     Round 1 (s→o): 3 pairs (last bLen=0). lN_=3.
//     Round 2 (o→s): 2 pairs (last bLen=0). lN_=2.
//     Round 3 (s→o): 1 pair. lN_=1. Break. No CopyBack.
//     Covers: 3-round merge, break after odd round, multiple odd pairs.
TEST_F(CscsortTest, WbMergeFiveRunsNoCopyBack)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    const int nnz = static_cast<int>(4U * runSize + 1U);
    const int m = nnz / 4 + 1;
    const int n = 1;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::mt19937 rng(44);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbMergeFiveRunsNoCopyBack"));
}

// Wb-5 RunSizeExactMultiple - len = 2*runSize, all runs full (no partial tail).
//     runCount=2, both runs exactly runSize. Phase1 line 254: remain == runSize_
//     → runLen = runSize_ (not the tail-remainder branch).
//     Covers: Phase1 full-run path (no tail run), 2-run merge with no partial.
TEST_F(CscsortTest, WbRunSizeExactMultiple)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    const int nnz = static_cast<int>(2U * runSize);
    const int m = nnz / 2 + 1;
    const int n = 1;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::mt19937 rng(45);
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbRunSizeExactMultiple"));
}

// Wb-6 AllSingleElemMultiCore - n > coreNum, every column has exactly 1 element.
//     All columns hit the len <= 1 skip (kernel line 115) in a multi-core
//     context. Each core enters the column loop but performs no sort work.
//     Covers: len==1 skip across multiple cores, multi-core with no-op kernel.
TEST_F(CscsortTest, WbAllSingleElemMultiCore)
{
    uint32_t coreNum = GetDeviceAivCoreCount();
    ASSERT_GT(coreNum, 0U);
    const int n = static_cast<int>(coreNum) + 10;
    const int nnz = n;
    const int m = nnz;
    std::vector<int32_t> colPtr(n + 1);
    std::vector<int32_t> rowInd(n);
    for (int j = 0; j < n; j++) {
        colPtr[j] = j;
        rowInd[j] = j % m;
    }
    colPtr[n] = n;
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbAllSingleElemMultiCore"));
}

// Wb-7 EmptyMatrixZeroDims - m=0, n=0, nnz=0.
//     Validation passes (m>=0, n>=0, nnz>=0; (m==0||n==0)&&nnz!=0 is false).
//     bufferSizeExt returns 0; sort returns SUCCESS without kernel launch.
//     Covers: host zero-dimension path distinct from EmptyMatrix (m=3,n=3,nnz=0).
TEST_F(CscsortTest, WbEmptyMatrixZeroDims)
{
    const int m = 0, n = 0, nnz = 0;
    std::vector<int32_t> colPtr = {0};
    std::vector<int32_t> rowInd;
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbEmptyMatrixZeroDims"));
}

// Wb-8 IdleCoresBetweenGaps - two non-empty columns separated by a large gap
//     of empty columns. Cores whose element range falls entirely within the
//     gap get colBegin == colEnd (for loop never executes). This exercises the
//     implicit idle-core path where FindColBoundary maps both elemBegin and
//     elemEnd to the same column boundary.
//     Covers: idle cores (colBegin==colEnd), FindColBoundary binary search
//     across empty gaps, multi-core with sparse column distribution.
TEST_F(CscsortTest, WbIdleCoresBetweenGaps)
{
    uint32_t coreNum = GetDeviceAivCoreCount();
    ASSERT_GT(coreNum, 0U);
    const int elemsPerCol = static_cast<int>(coreNum) * 4 + 10;
    const int n = 50;
    const int nnz = 2 * elemsPerCol;
    const int m = elemsPerCol;
    std::vector<int32_t> colPtr(n + 1, 0);
    colPtr[1] = elemsPerCol;      // col 0: elemsPerCol elements
    for (int j = 2; j < n; j++) {  // cols 1..n-2: empty
        colPtr[j] = elemsPerCol;
    }
    colPtr[n] = nnz;               // col n-1: elemsPerCol elements
    std::vector<int32_t> rowInd(nnz);
    for (int i = 0; i < elemsPerCol; i++) {
        rowInd[i] = (elemsPerCol - i) % m;
        rowInd[elemsPerCol + i] = i % m;
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbIdleCoresBetweenGaps"));
}

// Wb-9 TailCoreNonAligned - nnz not divisible by coreNum, last core gets a
//     partial element range. Combined with mixed column lengths (empty, short,
//     long) to exercise FindColBoundary for the tail core's non-aligned
//     element offsets.
//     Covers: tail core elemBegin/elemEnd (lines 107-108), FindColBoundary
//     for interior offsets, multi-core + multi-run on tail core.
TEST_F(CscsortTest, WbTailCoreNonAligned)
{
    uint32_t coreNum = GetDeviceAivCoreCount();
    ASSERT_GT(coreNum, 0U);
    // Choose n and per-column lengths so total nnz is NOT divisible by coreNum.
    // Use n slightly above coreNum with varying column lengths.
    const int n = static_cast<int>(coreNum) + 3;
    const int m = 100;
    std::vector<int32_t> colPtr(n + 1, 0);
    std::vector<int32_t> rowInd;
    std::mt19937 rng(88);
    for (int j = 0; j < n; j++) {
        int len;
        if (j % 3 == 0) {
            len = 0;              // empty column
        } else if (j % 3 == 1) {
            len = 1;              // single-element column
        } else {
            len = 5 + rng() % 10; // short column (2-14 elements)
        }
        for (int k = 0; k < len; k++) {
            rowInd.push_back(rng() % m);
        }
        colPtr[j + 1] = static_cast<int32_t>(rowInd.size());
    }
    int nnz = static_cast<int>(rowInd.size());
    // Verify nnz is not divisible by coreNum (if it happens to be, adjust)
    if (nnz % static_cast<int>(coreNum) == 0) {
        rowInd.push_back(0);
        colPtr[n] = static_cast<int32_t>(rowInd.size());
        nnz = static_cast<int>(rowInd.size());
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "WbTailCoreNonAligned"));
}

// ============================================================================
// 评审修订补充用例（R-1 / R-2 / R-3）
// ============================================================================

// R-1 PlaceholderMShortCols - m 为占位值 INT_MAX（torch hook 等调用方不给出
//     真实行数）时的短列批：行域上界由 kernel 批内实测（ReduceMax），不依赖 m。
TEST_F(CscsortTest, PlaceholderMShortCols)
{
    const int m = std::numeric_limits<int32_t>::max();
    const int n = 96;
    std::vector<int32_t> colPtr(n + 1, 0);
    std::vector<int32_t> rowInd;
    std::mt19937 rng(555);
    for (int j = 0; j < n; j++) {
        int len = 2 + static_cast<int>(rng() % 26);  // 2..27，全部落 32 槽位批
        for (int k = 0; k < len; k++) {
            rowInd.push_back(static_cast<int32_t>(rng() % 4096));
        }
        colPtr[j + 1] = static_cast<int32_t>(rowInd.size());
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(rowInd.size()), colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "PlaceholderMShortCols"));
}

// R-2 RowBeyondM - 行号大于 m-1（超出声明行域）：按值排序语义须保持（对标
//     cuSPARSE 按 cscRowInd 整数值稳定升序，与 m 无关）。col0-2 行号最大 100，
//     批内 ReduceMax 实测后走批量化矢量路径；col3（len=40，非 32 槽位）强制
//     FlushBatch 隔离前后两批；col4 含 2^20 行号，键宽超 24 位预算，回退标量
//     归并兜底。
TEST_F(CscsortTest, RowBeyondM)
{
    const int m = 8, n = 5, nnz = 55;
    std::vector<int32_t> colPtr = {0, 5, 9, 12, 52, 55};
    std::vector<int32_t> rowInd = {100, 3, 7, 0, 55, 2, 2, 100, 1, 6, 4, 42};
    for (int i = 0; i < 40; i++) {
        rowInd.push_back(static_cast<int32_t>((i * 3) % 8));  // col3：非 32 槽位分隔列
    }
    rowInd.push_back(7);
    rowInd.push_back(1048576);
    rowInd.push_back(3);
    // col0-2={100,3,7,0,55 | 2,2,100,1 | 6,4,42}：行号超 m，批内实测 maxRow=100
    //     → 批量化矢量路径（ReduceMax 自适应，按值正确排序）
    // col3：len=40 槽位 64≠32，FlushBatch 隔离前后两批（自身标量归并）
    // col4={7,1048576,3}：批内实测 2^20 超键宽预算 → 标量归并兜底
    EXPECT_TRUE(RunSortAndVerify(this, m, n, nnz, colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "RowBeyondM"));
}

// R-3 LongColNonAlignedMultiCore - len > runSize 的多 run 长列，起始元素偏移
//     非 32B 对齐（前列长度为奇数），且列总数跨多核：覆盖 GM 归并写回在列边界
//     不对齐时的 DataCopyPad 字节使能路径（标量 GM 写会与邻核 MTE3 互踩，见
//     kernel 文件头注释）。
TEST_F(CscsortTest, LongColNonAlignedMultiCore)
{
    uint32_t runSize = 0U;
    uint32_t sortTmpSize = 0U;
    ASSERT_TRUE(CscsortTiling::FindMaxRunSize(GetDeviceUbSize(), runSize, sortTmpSize));
    ASSERT_GT(runSize, 1U);
    uint32_t coreNum = GetDeviceAivCoreCount();
    ASSERT_GT(coreNum, 0U);
    const int m = 4096;
    std::vector<int32_t> colPtr;
    std::vector<int32_t> rowInd;
    std::mt19937 rng(2027);
    colPtr.push_back(0);
    auto pushCol = [&](int len) {
        for (int k = 0; k < len; k++) {
            rowInd.push_back(static_cast<int32_t>(rng() % m));
        }
        colPtr.push_back(static_cast<int32_t>(rowInd.size()));
    };
    pushCol(3);                                  // col0：3 元素，长列起始偏移 3（12B，非 32B 对齐）
    pushCol(static_cast<int>(runSize) + 37);     // col1：多 run 长列，非对齐起始
    const int n = static_cast<int>(coreNum) + 2;
    while (static_cast<int>(colPtr.size()) - 1 < n) {
        pushCol(1 + static_cast<int>(rng() % 31));  // 短列填充，保证多核有实际工作
    }
    EXPECT_TRUE(RunSortAndVerify(this, m, n, static_cast<int>(rowInd.size()), colPtr, rowInd,
                                 ACL_SPARSE_INDEX_BASE_ZERO, "LongColNonAlignedMultiCore"));
}

// ============================================================================
// Exception-path tests
// ============================================================================

TEST_F(CscsortTest, NullHandle)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(nullptr, 4, 4, 1, &dummy, &dummy, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseXcscsort(nullptr, 4, 4, 1, nullptr, &dummy, &dummy, &dummy, &bufSize),
        ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CscsortTest, NullBufferSize)
{
    int dummy = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 4, 4, 1, &dummy, &dummy, nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, InvalidM)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), -1, 4, 0, nullptr, nullptr, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, InvalidN)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 4, -1, 0, &dummy, nullptr, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, InvalidNnz)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 4, 4, -1, &dummy, &dummy, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, EmptyMatrixWithNnz)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 0, 4, 1, nullptr, &dummy, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, NullColPtr)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 4, 4, 1, nullptr, &dummy, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, NullRowInd)
{
    int dummy = 0;
    size_t bufSize = 0;
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 4, 4, 1, &dummy, nullptr, &bufSize),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, NullDescr)
{
    int dummy = 0;
    size_t bufSize = 0;
    std::vector<int32_t> colPtr = {0, 1};
    std::vector<int32_t> rowInd = {0};
    auto dColPtr = DeviceBuffer::copyFrom(colPtr.data(), 2 * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowInd.data(), sizeof(int32_t));
    EXPECT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 1, 1, 1,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<const int *>(dRowInd.raw()), &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    auto dBuf = DeviceBuffer::alloc(bufSize);
    auto dP = DeviceBuffer::alloc(sizeof(int32_t));
    EXPECT_EQ(aclsparseXcscsort(handle(), 1, 1, 1, nullptr,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<int *>(dRowInd.get()),
                  reinterpret_cast<int *>(dP.get()), dBuf.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(CscsortTest, NullP)
{
    std::vector<int32_t> colPtr = {0, 1};
    std::vector<int32_t> rowInd = {0};
    auto dColPtr = DeviceBuffer::copyFrom(colPtr.data(), 2 * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowInd.data(), sizeof(int32_t));
    size_t bufSize = 0;
    ASSERT_EQ(aclsparseXcscsort_bufferSizeExt(handle(), 1, 1, 1,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<const int *>(dRowInd.raw()), &bufSize),
        ACL_SPARSE_STATUS_SUCCESS);
    auto dBuf = DeviceBuffer::alloc(bufSize);
    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    EXPECT_EQ(aclsparseXcscsort(handle(), 1, 1, 1, descr,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<int *>(dRowInd.get()), nullptr, dBuf.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyMatDescr(descr);
}

TEST_F(CscsortTest, NullBuffer)
{
    std::vector<int32_t> colPtr = {0, 1};
    std::vector<int32_t> rowInd = {0};
    auto dColPtr = DeviceBuffer::copyFrom(colPtr.data(), 2 * sizeof(int32_t));
    auto dRowInd = DeviceBuffer::copyFrom(rowInd.data(), sizeof(int32_t));
    auto dP = DeviceBuffer::alloc(sizeof(int32_t));
    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    EXPECT_EQ(aclsparseXcscsort(handle(), 1, 1, 1, descr,
                  reinterpret_cast<const int *>(dColPtr.raw()),
                  reinterpret_cast<int *>(dRowInd.get()),
                  reinterpret_cast<int *>(dP.get()), nullptr),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    aclsparseDestroyMatDescr(descr);
}

// ============================================================================
// Performance benchmark tests (phase 4.1)
//
// ACL event based kernel-only timing for representative shape combinations.
// Run with: --gtest_filter=CscsortTest.PERF*
//
// Each case: warmup 3 iterations + measured 5 iterations, report median/min/max.
// Also reports tiling params (coreNum/runSize/simtThreads), estimated GM traffic
// and achieved bandwidth. Output is printed to stdout in a parseable format.
// ============================================================================

struct CscsortPerfResult {
    std::string name;
    int m = 0;
    int n = 0;
    int nnz = 0;
    uint32_t coreNum = 0;
    uint32_t aivTotal = 0;
    uint32_t runSize = 0;
    uint32_t simtThreads = 0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    uint64_t bytesMoved = 0;
    double bandwidthGbps = 0.0;  // GB/s
};

static double MedianDouble(std::vector<double> v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// Estimate GM bytes moved for a single cscsort launch.
// Counting rule (matches csrsort perf doc conservative estimate):
//   - Phase1: read rowInd+P (2*nnz*4B), write scratch rowInd+P (2*nnz*4B)
//   - Each merge round: read src (2*nnz*4B), write dst (2*nnz*4B)
//   - CopyBack (if final result in scratch): read scratch + write orig (4*nnz*4B)
// Uses the maximum per-column runCount to upper-bound merge rounds.
static uint64_t EstimateGmBytes(int nnz, uint32_t runSize,
    const std::vector<int32_t> &colPtrHost, int n)
{
    if (nnz <= 0 || runSize == 0) {
        return 0;
    }
    uint32_t maxColLen = 0;
    for (int col = 0; col < n; col++) {
        uint32_t len = static_cast<uint32_t>(colPtrHost[col + 1] - colPtrHost[col]);
        if (len > maxColLen) {
            maxColLen = len;
        }
    }
    uint32_t maxRunCount = (maxColLen + runSize - 1U) / runSize;
    if (maxRunCount <= 1U) {
        // single-run path: read rowInd+P, write rowInd+P
        return static_cast<uint64_t>(nnz) * 4ULL * 2ULL * 2ULL;
    }
    uint32_t mergeRounds = 0;
    uint32_t lN = maxRunCount;
    while (lN > 1U) {
        mergeRounds++;
        lN = (lN + 1U) / 2U;
    }
    // CopyBack happens when the final merge round writes to scratch.
    // Round 1: scratch->orig (FromScratch=true). Round 2: orig->scratch. Round 3: scratch->orig...
    // resultInScratch starts true (after Phase1 result is in scratch).
    // After round 1 (FromScratch=true): resultInScratch=false.
    // After round 2 (FromScratch=false): resultInScratch=true.
    // So CopyBack happens when mergeRounds is even (result ends in scratch).
    uint32_t copyback = (mergeRounds % 2U == 0U) ? 1U : 0U;
    // bytes = nnz * 4B * 2arrays * (Phase1:2 + mergeRounds*2 + copyback*2)
    return static_cast<uint64_t>(nnz) * 4ULL * 2ULL *
           static_cast<uint64_t>(2U + 2U * mergeRounds + 2U * copyback);
}

// Run a single perf case: warmup + measured iterations with ACL event timing.
static CscsortPerfResult RunPerfCase(CscsortTest *t, const std::string &name, int m, int n,
    const std::vector<int32_t> &colPtrHost, const std::vector<int32_t> &rowIndHost,
    int warmup = SparseTestFullEnabled() ? 3 : 1, int iters = SparseTestFullEnabled() ? 5 : 2)
{
    int nnz = static_cast<int>(rowIndHost.size());
    CscsortPerfResult r;
    r.name = name;
    r.m = m;
    r.n = n;
    r.nnz = nnz;

    uint64_t ubSize = GetDeviceUbSize();
    uint32_t runSize = 0U;
    uint32_t sortTmp = 0U;
    CscsortTiling::FindMaxRunSize(ubSize, runSize, sortTmp);
    uint32_t aivTotal = GetDeviceAivCoreCount();
    uint32_t taskUpper = std::min(static_cast<uint32_t>(n), static_cast<uint32_t>(nnz));
    uint32_t coreNum = std::min(aivTotal, taskUpper);
    r.runSize = runSize;
    r.simtThreads = 0U;  // arch22 无 SIMT，仅保留字段兼容 PERF 输出
    r.coreNum = coreNum;
    r.aivTotal = aivTotal;

    auto dColPtr = DeviceBuffer::copyFrom(colPtrHost.data(), static_cast<size_t>(n + 1) * sizeof(int32_t));
    size_t bufSize = 0;
    aclsparseXcscsort_bufferSizeExt(t->handle(), m, n, nnz,
        reinterpret_cast<const int *>(dColPtr.raw()),
        reinterpret_cast<const int *>(dColPtr.raw()),  // dummy, only size is needed
        &bufSize);
    if (nnz == 0 || bufSize == 0) {
        r.bytesMoved = 0;
        r.bandwidthGbps = 0.0;
        return r;
    }

    auto dBuf = DeviceBuffer::alloc(bufSize);
    auto dRowInd = DeviceBuffer::alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
    auto dP = DeviceBuffer::alloc(static_cast<size_t>(nnz) * sizeof(int32_t));
    std::vector<int32_t> pHost(nnz);
    std::iota(pHost.begin(), pHost.end(), 0);

    aclsparseMatDescr_t descr = nullptr;
    aclsparseCreateMatDescr(&descr);
    aclsparseSetMatIndexBase(descr, ACL_SPARSE_INDEX_BASE_ZERO);

    auto runOnce = [&]() {
        aclrtMemcpy(dRowInd.raw(), static_cast<size_t>(nnz) * sizeof(int32_t), rowIndHost.data(),
                    static_cast<size_t>(nnz) * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(dP.raw(), static_cast<size_t>(nnz) * sizeof(int32_t), pHost.data(),
                    static_cast<size_t>(nnz) * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        aclsparseXcscsort(t->handle(), m, n, nnz, descr,
            reinterpret_cast<const int *>(dColPtr.raw()),
            reinterpret_cast<int *>(dRowInd.get()),
            reinterpret_cast<int *>(dP.get()), dBuf.get());
    };

    // warmup
    for (int i = 0; i < warmup; i++) {
        runOnce();
    }
    aclrtSynchronizeStream(t->stream());

    // measured iterations
    std::vector<double> times;
    times.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; i++) {
        aclrtEvent start = nullptr;
        aclrtEvent stop = nullptr;
        aclrtCreateEvent(&start);
        aclrtCreateEvent(&stop);
        aclrtRecordEvent(start, t->stream());
        runOnce();
        aclrtRecordEvent(stop, t->stream());
        aclrtSynchronizeStream(t->stream());
        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, start, stop);
        times.push_back(static_cast<double>(ms));
        aclrtDestroyEvent(start);
        aclrtDestroyEvent(stop);
    }
    aclsparseDestroyMatDescr(descr);

    r.minMs = *std::min_element(times.begin(), times.end());
    r.maxMs = *std::max_element(times.begin(), times.end());
    r.medianMs = MedianDouble(times);

    r.bytesMoved = EstimateGmBytes(nnz, runSize, colPtrHost, n);
    if (r.medianMs > 0.0) {
        // GB/s = bytes / (s) = bytes / (ms * 1e-3) / 1e9 = bytes / (ms * 1e6)
        r.bandwidthGbps = static_cast<double>(r.bytesMoved) / (r.medianMs * 1e6);
    }
    return r;
}

static void PrintPerfResult(const CscsortPerfResult &r)
{
    std::printf("[PERF] %-26s m=%-5d n=%-5d nnz=%-8d core=%u/%u runSize=%u simtThreads=%u "
                "median=%.4fms min=%.4fms max=%.4fms bytes=%llu bw=%.2fGB/s\n",
                r.name.c_str(), r.m, r.n, r.nnz, r.coreNum, r.aivTotal, r.runSize, r.simtThreads,
                r.medianMs, r.minMs, r.maxMs,
                static_cast<unsigned long long>(r.bytesMoved), r.bandwidthGbps);
    std::fflush(stdout);
}

// Build a CSC matrix with uniform per-column length `elemsPerCol` (last col absorbs remainder).
static void BuildUniformCsc(int m, int n, int elemsPerCol, uint32_t seed,
    std::vector<int32_t> &colPtr, std::vector<int32_t> &rowInd)
{
    std::mt19937 rng(seed);
    colPtr.assign(static_cast<size_t>(n) + 1, 0);
    rowInd.clear();
    for (int col = 0; col < n; col++) {
        for (int k = 0; k < elemsPerCol; k++) {
            rowInd.push_back(static_cast<int32_t>(rng() % m));
        }
        colPtr[col + 1] = static_cast<int32_t>(rowInd.size());
    }
}

// Build a CSC matrix with skewed column lengths: few long columns, many short/empty.
static void BuildSkewedCsc(int m, int n, int longColLen, int shortColLen, int longColEvery,
    uint32_t seed, std::vector<int32_t> &colPtr, std::vector<int32_t> &rowInd)
{
    std::mt19937 rng(seed);
    colPtr.assign(static_cast<size_t>(n) + 1, 0);
    rowInd.clear();
    for (int col = 0; col < n; col++) {
        int len;
        if (col % longColEvery == 0) {
            len = longColLen;
        } else if (col % 4 == 0) {
            len = shortColLen;
        } else {
            len = col % 5;  // 0..4, often empty or single
        }
        for (int k = 0; k < len; k++) {
            rowInd.push_back(static_cast<int32_t>(rng() % m));
        }
        colPtr[col + 1] = static_cast<int32_t>(rowInd.size());
    }
}

// PERF-1 SmallScale - 100x100, ~1000 nnz, uniform
TEST_F(CscsortTest, PERF_SmallScale)
{
    const int m = 100, n = 100, elemsPerCol = 10;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 11, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_SmallScale", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-2 MediumScale - 1000x1000, ~10000 nnz, uniform
TEST_F(CscsortTest, PERF_MediumScale)
{
    const int m = SparseTestFullEnabled() ? 1000 : 128;
    const int n = SparseTestFullEnabled() ? 1000 : 128;
    const int elemsPerCol = SparseTestFullEnabled() ? 10 : 4;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 22, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_MediumScale", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-3 LargeScale - 4096x4096, ~100000 nnz, uniform
TEST_F(CscsortTest, PERF_LargeScale)
{
    const int m = SparseTestFullEnabled() ? 4096 : 256;
    const int n = SparseTestFullEnabled() ? 4096 : 256;
    const int elemsPerCol = SparseTestFullEnabled() ? 24 : 4;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 33, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_LargeScale", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-4 MillionNnz - 4096x4096, ~1000000 nnz, uniform (64 cols x 16384 elems, mirrors
// the LargeNnzMultiCore functional case for direct csrsort comparison)
TEST_F(CscsortTest, PERF_MillionNnz)
{
    const int m = SparseTestFullEnabled() ? 4096 : 256;
    const int n = SparseTestFullEnabled() ? 64 : 16;
    const int elemsPerCol = SparseTestFullEnabled() ? 16384 : 256;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 44, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_MillionNnz", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-5 SkewedCols - 1000x1000, ~10000 nnz, skewed (few long columns)
TEST_F(CscsortTest, PERF_SkewedCols)
{
    const int m = SparseTestFullEnabled() ? 1000 : 128;
    const int n = SparseTestFullEnabled() ? 1000 : 128;
    std::vector<int32_t> colPtr, rowInd;
    BuildSkewedCsc(m, n, SparseTestFullEnabled() ? 10000 : 256,
                   SparseTestFullEnabled() ? 2048 : 32, /*longColEvery=*/64,
                   55, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_SkewedCols", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-6 SingleLongCol - 1x1 matrix, single column with 32768 elements (multi-run merge path,
// mirrors LongColMultiRun for direct csrsort comparison)
TEST_F(CscsortTest, PERF_SingleLongCol)
{
    const int m = SparseTestFullEnabled() ? 4096 : 256;
    const int n = 1;
    const int nnz = SparseTestFullEnabled() ? 32768 : 2048;
    std::vector<int32_t> colPtr = {0, nnz};
    std::vector<int32_t> rowInd(nnz);
    std::mt19937 rng(2026);
    for (int i = 0; i < nnz; i++) {
        rowInd[i] = i % m;
    }
    std::shuffle(rowInd.begin(), rowInd.end(), rng);
    auto r = RunPerfCase(this, "PERF_SingleLongCol", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-7 MultiCoreSmallCols - 100x1000, ~3000 nnz, 3 elems per col (multi-core parallelism)
TEST_F(CscsortTest, PERF_MultiCoreSmallCols)
{
    const int m = 100;
    const int n = SparseTestFullEnabled() ? 1000 : 128;
    const int elemsPerCol = 3;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 77, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_MultiCoreSmallCols", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-A P01 - 验收锚点：Llama 3.1 70B MLP 维度 CSC 8192x28672，nnz=524288（约18/列）
// 标杆 GPU median 2961.7-2969.3us，0.25x 门槛即 NPU <= 约11.8ms
TEST_F(CscsortTest, PERF_P01_Llama70B)
{
    const int m = SparseTestFullEnabled() ? 8192 : 256;
    const int n = SparseTestFullEnabled() ? 28672 : 256;
    const int elemsPerCol = SparseTestFullEnabled() ? 18 : 4;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 101, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_P01_Llama70B", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-A P02 - 验收锚点：Qwen3-235B MoE 维度 CSC 4096x1536，nnz=262144（约170/列）
// 标杆 GPU median 2595.9-2610.4us，0.25x 门槛即 NPU <= 约10.4ms
TEST_F(CscsortTest, PERF_P02_Qwen235B)
{
    const int m = SparseTestFullEnabled() ? 4096 : 256;
    const int n = SparseTestFullEnabled() ? 1536 : 128;
    const int elemsPerCol = SparseTestFullEnabled() ? 170 : 8;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 102, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_P02_Qwen235B", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// PERF-A P03 - 验收锚点：DeepSeek-V3 MoE 维度 CSC 7168x2048，nnz=458752（224/列）
// 标杆 GPU median 2780.8-2783.1us，0.25x 门槛即 NPU <= 约11.1ms
TEST_F(CscsortTest, PERF_P03_DeepSeekV3)
{
    const int m = SparseTestFullEnabled() ? 7168 : 256;
    const int n = SparseTestFullEnabled() ? 2048 : 128;
    const int elemsPerCol = SparseTestFullEnabled() ? 224 : 8;
    std::vector<int32_t> colPtr, rowInd;
    BuildUniformCsc(m, n, elemsPerCol, 103, colPtr, rowInd);
    auto r = RunPerfCase(this, "PERF_P03_DeepSeekV3", m, n, colPtr, rowInd);
    PrintPerfResult(r);
    SUCCEED();
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    g_acl_env = new AclTestEnvironment();
    testing::AddGlobalTestEnvironment(g_acl_env);
    return RUN_ALL_TESTS();
}
