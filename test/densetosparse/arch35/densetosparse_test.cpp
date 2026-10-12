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

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "aclsparse_descr_internal.h"
#include "densetosparse_golden.h"
#include "densetosparse_kernel.h"
#include "densetosparse_npu_wrapper.h"
#include "densetosparse_param.h"
#include "sparse_test.h"

using namespace sparse_test;

namespace {

template <typename Base>
class DenseToSparseAclEnvironment : public Base {
public:
  static void SetUpTestSuite() {
    env_ = std::make_unique<AclEnvScope>();
    handle_ = std::make_unique<HandleManager>();
    handle_->setStream(env_->stream());
  }
  static void TearDownTestSuite() {
    handle_.reset();
    env_.reset();
  }

protected:
  inline static std::unique_ptr<AclEnvScope> env_;
  inline static std::unique_ptr<HandleManager> handle_;
};

template <typename Param>
class DenseToSparseParamTest
    : public DenseToSparseAclEnvironment<testing::TestWithParam<Param>> {};

using DenseToSparseTest = DenseToSparseParamTest<DenseToSparseParam>;

void ExpectDenseToSparseStatuses(const DenseToSparseParam &p,
                                 const DenseToSparseRunResult &actual) {
  ASSERT_EQ(actual.descriptorStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
  ASSERT_EQ(actual.queryStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
  ASSERT_EQ(actual.analysisStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
  ASSERT_EQ(actual.setPointersStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
  ASSERT_EQ(actual.convertStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
  ASSERT_EQ(actual.syncStatus, ACL_SUCCESS) << p.caseId();
  if (p.m == 0 || p.n == 0 || p.format == "BELL") {
    EXPECT_EQ(actual.workspaceSize, 0u) << p.caseId();
  } else {
    EXPECT_GT(actual.workspaceSize, 0u) << p.caseId();
  }
  if (p.distribution == "WORKSPACE_OFFSET") {
    EXPECT_NE(actual.analysisWorkspaceAddress, 0u) << p.caseId();
    EXPECT_NE(actual.convertWorkspaceAddress, 0u) << p.caseId();
    EXPECT_NE(actual.analysisWorkspaceAddress, actual.convertWorkspaceAddress)
        << p.caseId();
  }
}

void ExpectDenseToSparseGolden(const DenseToSparseParam &p,
                               const DenseToSparseGoldenResult &golden,
                               const DenseToSparseRunResult &actual) {
  if (p.format == "BELL") {
    EXPECT_EQ(actual.bellNnzAfterQuery, golden.genericNnz);
    EXPECT_EQ(actual.bellNnzAfterAnalysis, golden.genericNnz);
    EXPECT_EQ(actual.bellNnzAfterConvert, golden.genericNnz);
    EXPECT_EQ(actual.ellPattern,
              PackDenseToSparseIndices(golden.ellPattern,
                                       DenseToSparseIndexType(p.index_type)));
  } else {
    EXPECT_EQ(actual.queriedNnz, golden.nnz);
    EXPECT_EQ(actual.offsets,
              PackDenseToSparseIndices(golden.offsets,
                                       DenseToSparseIndexType(p.offset_type)));
    EXPECT_EQ(actual.indices0,
              PackDenseToSparseIndices(golden.indices0,
                                       DenseToSparseIndexType(p.index_type)));
    EXPECT_EQ(actual.indices1,
              PackDenseToSparseIndices(golden.indices1,
                                       DenseToSparseIndexType(p.index_type)));
  }
  EXPECT_EQ(actual.values, golden.values) << p.caseId();
}

TEST_P(DenseToSparseTest, ThreeStageBitwiseGolden) {
  const auto &p = GetParam();
  if (p.descriptorOnly() || p.bufferSizeOnly()) {
    GTEST_SKIP() << "safe theoretical boundary is covered by descriptor tests";
  }
  const auto host = MakeDenseToSparseInput(p);
  const size_t width =
      DenseToSparseTypeSize(DenseToSparseValueType(p.value_type));
  const auto golden = DenseToSparseGolden(
      DenseToSparseFormatOf(p.format), p.m, p.n, p.ld, p.order == "ROW",
      p.base == "ONE" ? 1 : 0, width, p.value_type == "INT8", host.dense,
      p.block_size, p.ell_cols, host.ellPattern);
  std::vector<uint8_t> firstValues;
  for (int run = 0; run < p.repeat; ++run) {
    const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
    ExpectDenseToSparseStatuses(p, actual);
    ExpectDenseToSparseGolden(p, golden, actual);
    if (run == 0) {
      firstValues = actual.values;
    } else {
      EXPECT_EQ(actual.values, firstValues) << "non-deterministic output";
    }
  }
}

std::vector<DenseToSparseParam> LoadDenseToSparseCases() {
  auto cases = GetCasesFromCsv<DenseToSparseParam>("densetosparse_test.csv");
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [](const DenseToSparseParam &p) {
                               return p.descriptorOnly() ||
                                      p.bufferSizeOnly();
                             }),
              cases.end());
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    CsvL0L1, DenseToSparseTest,
    testing::ValuesIn(LoadDenseToSparseCases()),
    [](const testing::TestParamInfo<DenseToSparseParam> &info) {
      std::string name = info.param.caseId();
      std::replace_if(
          name.begin(), name.end(),
          [](char c) {
            return !(std::isalnum(static_cast<unsigned char>(c)) || c == '_');
          },
          '_');
      return name;
    });

using DenseToSparseExceptionTest =
    DenseToSparseAclEnvironment<testing::Test>;

DenseToSparseParam MakeStatelessParam(const std::string &format) {
  DenseToSparseParam p;
  p.case_name = "stateless_" + format;
  p.level = "L1";
  p.mode = "execute";
  p.format = format;
  p.offset_type = "I32";
  p.index_type = "I32";
  p.base = "ZERO";
  p.value_type = "FP32";
  p.order = "ROW";
  p.distribution = format == "BELL" ? "BELL_FULL" : "Z50";
  p.m = 4;
  p.n = 4;
  p.ld = 4;
  p.block_size = format == "BELL" ? 2 : 1;
  p.ell_cols = format == "BELL" ? 4 : 0;
  p.seed = 701;
  return p;
}

std::vector<uint8_t> PackFp32Bits(const std::vector<uint32_t> &bits) {
  std::vector<uint8_t> bytes(bits.size() * sizeof(uint32_t));
  const auto *begin = reinterpret_cast<const uint8_t *>(bits.data());
  std::copy(begin, begin + bytes.size(), bytes.begin());
  return bytes;
}

TEST_F(DenseToSparseExceptionTest,
       CsrExcludesSignedZerosAndPreservesSpecialBits) {
  auto p = MakeStatelessParam("CSR");
  p.m = 2;
  p.n = 3;
  p.ld = 3;
  DenseToSparseHostInput host;
  host.dense = PackFp32Bits({0x00000000u, 0x80000000u, 0x7fc12345u,
                             0x7f800000u, 0xff800000u, 0x3fc00000u});
  const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
  ExpectDenseToSparseStatuses(p, actual);
  EXPECT_EQ(actual.queriedNnz, 4);
  EXPECT_EQ(actual.offsets,
            PackDenseToSparseIndices({0, 1, 4}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.indices0,
            PackDenseToSparseIndices({2, 0, 1, 2}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.values,
            PackFp32Bits(
                {0x7fc12345u, 0x7f800000u, 0xff800000u, 0x3fc00000u}));
}

TEST_F(DenseToSparseExceptionTest, ColumnMajorLdPaddingIsNotScanned) {
  auto p = MakeStatelessParam("CSC");
  p.base = "ONE";
  p.order = "COL";
  p.m = 2;
  p.n = 2;
  p.ld = 3;
  DenseToSparseHostInput host;
  host.dense = PackFp32Bits({0x3f800000u, 0x00000000u, 0x7fc00001u,
                             0x40000000u, 0x40400000u, 0x7fc00002u});
  const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
  ExpectDenseToSparseStatuses(p, actual);
  EXPECT_EQ(actual.queriedNnz, 3);
  EXPECT_EQ(actual.offsets,
            PackDenseToSparseIndices({1, 2, 4}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.indices0,
            PackDenseToSparseIndices({1, 1, 2}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.values,
            PackFp32Bits({0x3f800000u, 0x40000000u, 0x40400000u}));
}

TEST_F(DenseToSparseExceptionTest,
       BlockedEllPreservesPatternAndUsesPositiveZeroPadding) {
  auto p = MakeStatelessParam("BELL");
  DenseToSparseHostInput host;
  host.dense = PackFp32Bits(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  host.ellPattern = {0, -1, 1, -1};
  const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
  ExpectDenseToSparseStatuses(p, actual);
  EXPECT_EQ(actual.bellNnzAfterQuery, 16);
  EXPECT_EQ(actual.bellNnzAfterAnalysis, 16);
  EXPECT_EQ(actual.bellNnzAfterConvert, 16);
  EXPECT_EQ(actual.ellPattern,
            PackDenseToSparseIndices(host.ellPattern, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.values,
            PackFp32Bits(
                {1, 5, 2, 6, 0, 0, 0, 0, 11, 15, 12, 16, 0, 0, 0, 0}));
}

std::vector<uint8_t> PackComplex64Bits(const std::vector<uint64_t> &bits) {
  std::vector<uint8_t> bytes(bits.size() * sizeof(uint64_t));
  const auto *begin = reinterpret_cast<const uint8_t *>(bits.data());
  std::copy(begin, begin + bytes.size(), bytes.begin());
  return bytes;
}

// complex64 is stored as (imaginary << 32) | real.
constexpr uint64_t kC64Zero = 0x0000000000000000ull;
constexpr uint64_t kC64NegZeroPair = 0x8000000080000000ull;
constexpr uint64_t kC64RealOnly = 0x000000003fc00000ull;
constexpr uint64_t kC64ImagOnly = 0xbfc0000000000000ull;
constexpr uint64_t kC64NanImag = 0x7fc1234500000000ull;
constexpr uint64_t kC64InfPair = 0xff8000007f800000ull;

TEST_F(DenseToSparseExceptionTest, Complex64KeepsEitherComponentNonzero) {
  auto p = MakeStatelessParam("CSR");
  p.value_type = "CPLX64";
  p.m = 2;
  p.n = 3;
  p.ld = 3;
  DenseToSparseHostInput host;
  host.dense = PackComplex64Bits({kC64Zero, kC64NegZeroPair, kC64RealOnly,
                                  kC64ImagOnly, kC64NanImag, kC64InfPair});
  const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
  ExpectDenseToSparseStatuses(p, actual);
  EXPECT_EQ(actual.queriedNnz, 4);
  EXPECT_EQ(actual.offsets,
            PackDenseToSparseIndices({0, 1, 4}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.indices0,
            PackDenseToSparseIndices({2, 0, 1, 2}, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actual.values,
            PackComplex64Bits(
                {kC64RealOnly, kC64ImagOnly, kC64NanImag, kC64InfPair}));
}

TEST_F(DenseToSparseExceptionTest, Complex64MatchesGoldenAcrossFormats) {
  for (const std::string format : {"CSR", "CSC", "COO"}) {
    auto p = MakeStatelessParam(format);
    p.value_type = "CPLX64";
    p.base = "ONE";
    p.distribution = "SPECIAL";
    const auto host = MakeDenseToSparseInput(p);
    const auto golden = DenseToSparseGolden(DenseToSparseFormatOf(p.format),
                                            p.m, p.n, p.ld, true, 1,
                                            sizeof(uint64_t), false, host.dense);
    const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
    ExpectDenseToSparseStatuses(p, actual);
    ExpectDenseToSparseGolden(p, golden, actual);
  }
}

TEST_F(DenseToSparseExceptionTest, BlockedEllTailBlocksPadWithPositiveZero) {
  auto p = MakeStatelessParam("BELL");
  p.m = 3;
  p.n = 3;
  p.ld = 3;
  p.block_size = 2;
  p.ell_cols = 2;
  DenseToSparseHostInput host;
  host.dense = PackFp32Bits({1, 2, 3, 4, 5, 6, 7, 8, 9});
  host.ellPattern = {0, 1};
  const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
  ExpectDenseToSparseStatuses(p, actual);
  // 2 block rows x 1 slot, each owning a full 2x2 block, so 8 scalars.
  EXPECT_EQ(actual.bellNnzAfterQuery, 8);
  EXPECT_EQ(actual.bellNnzAfterAnalysis, 8);
  EXPECT_EQ(actual.bellNnzAfterConvert, 8);
  EXPECT_EQ(actual.ellPattern,
            PackDenseToSparseIndices(host.ellPattern, ACL_SPARSE_INDEX_32I));
  // Block (0,0) is whole; block (1,1) only has row 2 and column 2 in range.
  EXPECT_EQ(actual.values, PackFp32Bits({1, 4, 2, 5, 9, 0, 0, 0}));
}

TEST_F(DenseToSparseExceptionTest, BlockedEllTailMatchesGoldenForAllBlockSizes) {
  for (const int64_t blockSize : {2, 4}) {
    for (const std::string order : {"ROW", "COL"}) {
      auto p = MakeStatelessParam("BELL");
      p.order = order;
      p.base = "ONE";
      p.m = 7;
      p.n = 7;
      p.ld = 9;
      p.block_size = blockSize;
      p.ell_cols = 5;
      p.distribution = "BELL_FULL";
      const auto host = MakeDenseToSparseInput(p);
      const auto golden = DenseToSparseGolden(
          DenseSparseFormat::BLOCKED_ELL, p.m, p.n, p.ld, order == "ROW", 1,
          sizeof(uint32_t), false, host.dense, p.block_size, p.ell_cols,
          host.ellPattern);
      const auto actual = RunDenseToSparse(*handle_, env_->stream(), p, host);
      ExpectDenseToSparseStatuses(p, actual);
      ExpectDenseToSparseGolden(p, golden, actual);
    }
  }
}

TEST_F(DenseToSparseExceptionTest, GetBufferSizeIsRepeatablePureQuery) {
  const auto p = MakeStatelessParam("CSR");
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(*handle_, env_->stream(), p, host);
  ASSERT_TRUE(PrepareDenseToSparseRun(&ctx));
  auto *standard = reinterpret_cast<aclsparseSpMatDescr *>(ctx.sparse.get());
  const auto nnz = standard->nnz;
  const auto offsets = standard->ptrs;
  const auto indices = standard->idxs;
  const auto values = standard->values;
  size_t first = 0;
  size_t second = 0;
  ASSERT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &first),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &second),
            ACL_SPARSE_STATUS_SUCCESS);
  EXPECT_EQ(first, second);
  EXPECT_EQ(standard->nnz, nnz);
  EXPECT_EQ(standard->ptrs, offsets);
  EXPECT_EQ(standard->idxs, indices);
  EXPECT_EQ(standard->values, values);
}

TEST_F(DenseToSparseExceptionTest,
       AnalysisWithoutQueryPublishesStandardMetadataForAllFormats) {
  for (const std::string format : {"CSR", "CSC", "COO", "BELL"}) {
    const auto p = MakeStatelessParam(format);
    const auto host = MakeDenseToSparseInput(p);
    DenseToSparseRunContext sizing(*handle_, env_->stream(), p, host);
    ASSERT_TRUE(PrepareDenseToSparseRun(&sizing)) << format;
    size_t workspaceSize = 0;
    ASSERT_EQ(aclsparseDenseToSparseGetBufferSize(
                  handle_->get(), sizing.dense.get(), sizing.sparse.get(),
                  ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &workspaceSize),
              ACL_SPARSE_STATUS_SUCCESS)
        << format;

    DenseToSparseRunContext target(*handle_, env_->stream(), p, host);
    ASSERT_TRUE(PrepareDenseToSparseRun(&target)) << format;
    if (workspaceSize != 0) {
      target.analysisWorkspace =
          std::make_unique<DeviceBuffer>(DeviceBuffer::alloc(workspaceSize));
    }
    ASSERT_EQ(aclsparseDenseToSparseAnalysis(
                  handle_->get(), target.dense.get(), target.sparse.get(),
                  ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                  target.analysisWorkspace ? target.analysisWorkspace->get()
                                           : nullptr),
              ACL_SPARSE_STATUS_SUCCESS)
        << format;
    ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS) << format;

    const auto golden = DenseToSparseGolden(
        DenseToSparseFormatOf(p.format), p.m, p.n, p.ld, true, 0,
        sizeof(uint32_t), false, host.dense, p.block_size, p.ell_cols,
        host.ellPattern);
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = -1;
    ASSERT_EQ(aclsparseSpMatGetSize(target.sparse.get(), &rows, &cols, &nnz),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(nnz, format == "BELL" ? golden.genericNnz : golden.nnz)
        << format;
    if (format == "CSR" || format == "CSC") {
      std::vector<uint8_t> actual(golden.offsets.size() * sizeof(int32_t));
      target.offsets->copyToHost(actual.data(), actual.size());
      EXPECT_EQ(actual, PackDenseToSparseIndices(
                            golden.offsets, ACL_SPARSE_INDEX_32I))
          << format;
    }
  }
}

TEST_F(DenseToSparseExceptionTest,
       ConvertUsesCurrentDenseAndReboundOutputPointers) {
  const auto p = MakeStatelessParam("CSR");
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(*handle_, env_->stream(), p, host);
  ASSERT_TRUE(PrepareDenseToSparseRun(&ctx));
  ASSERT_TRUE(QueryAndAnalyzeDenseToSparse(&ctx));
  ASSERT_TRUE(BindDenseToSparseRunOutput(&ctx));

  std::vector<uint8_t> analyzedOffsets(ctx.offsetCount * ctx.offsetWidth);
  ctx.offsets->copyToHost(analyzedOffsets.data(), analyzedOffsets.size());
  auto replacementOffsets =
      DeviceBuffer::copyFrom(analyzedOffsets.data(), analyzedOffsets.size());
  auto replacementIndices = DeviceBuffer::alloc(
      static_cast<size_t>(ctx.result.queriedNnz) * ctx.indexWidth);
  auto replacementValues = DeviceBuffer::alloc(
      static_cast<size_t>(ctx.result.queriedNnz) * ctx.valueWidth);
  ASSERT_EQ(aclsparseCsrSetPointers(ctx.sparse.get(), replacementOffsets.get(),
                                   replacementIndices.get(),
                                   replacementValues.get()),
            ACL_SPARSE_STATUS_SUCCESS);

  auto replacementDense =
      DeviceBuffer::copyFrom(host.dense.data(), host.dense.size());
  DenseToSparseConstDnMat currentDense;
  ASSERT_EQ(currentDense.create(p.m, p.n, p.ld, replacementDense.get(),
                                ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseDenseToSparseConvert(
                handle_->get(), currentDense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                ctx.convertWorkspace->get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto golden = DenseToSparseGolden(
      DenseToSparseFormatOf(p.format), p.m, p.n, p.ld, true, 0,
      sizeof(uint32_t), false, host.dense, p.block_size, p.ell_cols,
      host.ellPattern);
  std::vector<uint8_t> actualIndices(golden.indices0.size() * ctx.indexWidth);
  std::vector<uint8_t> actualValues(golden.values.size());
  replacementIndices.copyToHost(actualIndices.data(), actualIndices.size());
  replacementValues.copyToHost(actualValues.data(), actualValues.size());
  EXPECT_EQ(actualIndices, PackDenseToSparseIndices(
                               golden.indices0, ACL_SPARSE_INDEX_32I));
  EXPECT_EQ(actualValues, golden.values);
}

::testing::AssertionResult
PrepareNnzMismatchRun(DenseToSparseRunContext *ctx,
                      std::vector<uint8_t> *indexSentinel,
                      std::vector<uint8_t> *valueSentinel) {
  if (!PrepareDenseToSparseRun(ctx))
    return ::testing::AssertionFailure() << "failed to prepare run";
  if (!QueryAndAnalyzeDenseToSparse(ctx))
    return ::testing::AssertionFailure() << "failed to query and analyze";
  if (ctx->result.queriedNnz <= 1)
    return ::testing::AssertionFailure()
           << "queried nnz must exceed 1, got " << ctx->result.queriedNnz;
  auto *standard =
      reinterpret_cast<aclsparseSpMatDescr *>(ctx->sparse.get());
  --standard->nnz;
  indexSentinel->assign(
      static_cast<size_t>(ctx->result.queriedNnz) * ctx->indexWidth, 0xA5);
  valueSentinel->assign(
      static_cast<size_t>(ctx->result.queriedNnz) * ctx->valueWidth, 0x5A);
  return ::testing::AssertionSuccess();
}

TEST_F(DenseToSparseExceptionTest, ConvertNnzMismatchDoesNotWriteOutput) {
  const auto p = MakeStatelessParam("CSR");
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(*handle_, env_->stream(), p, host);
  std::vector<uint8_t> indexSentinel;
  std::vector<uint8_t> valueSentinel;
  ASSERT_TRUE(PrepareNnzMismatchRun(&ctx, &indexSentinel, &valueSentinel));
  auto indices =
      DeviceBuffer::copyFrom(indexSentinel.data(), indexSentinel.size());
  auto values = DeviceBuffer::copyFrom(valueSentinel.data(),
                                       valueSentinel.size());
  ASSERT_EQ(aclsparseCsrSetPointers(ctx.sparse.get(), ctx.offsets->get(),
                                   indices.get(), values.get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseDenseToSparseConvert(
                handle_->get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                ctx.convertWorkspace->get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  std::vector<uint8_t> actualIndices(indexSentinel.size());
  std::vector<uint8_t> actualValues(valueSentinel.size());
  indices.copyToHost(actualIndices.data(), actualIndices.size());
  values.copyToHost(actualValues.data(), actualValues.size());
  EXPECT_EQ(actualIndices, indexSentinel);
  EXPECT_EQ(actualValues, valueSentinel);
}

struct DenseToSparseOffsetCase {
  const char *format;
  const char *offsetType;
  int mutation;
};

void ExpectDeviceBuffersEqual(DeviceBuffer &first,
                              const std::vector<uint8_t> &firstExpected,
                              DeviceBuffer &second,
                              const std::vector<uint8_t> &secondExpected) {
  std::vector<uint8_t> firstActual(firstExpected.size());
  std::vector<uint8_t> secondActual(secondExpected.size());
  first.copyToHost(firstActual.data(), firstActual.size());
  second.copyToHost(secondActual.data(), secondActual.size());
  EXPECT_EQ(firstActual, firstExpected);
  EXPECT_EQ(secondActual, secondExpected);
}

void ExpectCompressedOffsetErrorDoesNotWrite(
    HandleManager &handle, aclrtStream stream,
    const DenseToSparseOffsetCase &item) {
  auto p = MakeStatelessParam(item.format);
  p.offset_type = item.offsetType;
  p.index_type = item.offsetType;
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(handle, stream, p, host);
  ASSERT_TRUE(PrepareDenseToSparseRun(&ctx)) << item.format;
  ASSERT_TRUE(QueryAndAnalyzeDenseToSparse(&ctx)) << item.format;
  auto offsets =
      DenseToSparseGolden(DenseToSparseFormatOf(p.format), p.m, p.n, p.ld,
                          true, 0, sizeof(uint32_t), false, host.dense,
                          p.block_size, p.ell_cols, host.ellPattern)
          .offsets;
  if (item.mutation == 0)
    offsets.front() = 1;
  else if (item.mutation == 1)
    offsets[2] = offsets[1] - 1;
  else
    offsets.back() = ctx.result.queriedNnz + 1;
  const auto packedOffsets = PackDenseToSparseIndices(
      offsets, DenseToSparseIndexType(p.offset_type));
  auto deviceOffsets =
      DeviceBuffer::copyFrom(packedOffsets.data(), packedOffsets.size());
  const std::vector<uint8_t> indexSentinel(
      static_cast<size_t>(ctx.result.queriedNnz) * ctx.indexWidth, 0xA5);
  const std::vector<uint8_t> valueSentinel(
      static_cast<size_t>(ctx.result.queriedNnz) * ctx.valueWidth, 0x5A);
  auto indices =
           DeviceBuffer::copyFrom(indexSentinel.data(), indexSentinel.size()),
       values =
           DeviceBuffer::copyFrom(valueSentinel.data(), valueSentinel.size());
  ASSERT_EQ(p.format == "CSR"
                ? aclsparseCsrSetPointers(ctx.sparse.get(), deviceOffsets.get(),
                                          indices.get(), values.get())
                : aclsparseCscSetPointers(ctx.sparse.get(), deviceOffsets.get(),
                                          indices.get(), values.get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseDenseToSparseConvert(
                handle.get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                ctx.convertWorkspace->get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  ExpectDeviceBuffersEqual(indices, indexSentinel, values, valueSentinel);
}

TEST_F(DenseToSparseExceptionTest,
       CompressedOffsetErrorsDoNotWriteOutput) {
  const DenseToSparseOffsetCase cases[] = {
      {"CSR", "I32", 0}, {"CSR", "I64", 1}, {"CSC", "I32", 2}};
  for (const auto &item : cases)
    ExpectCompressedOffsetErrorDoesNotWrite(*handle_, env_->stream(), item);
}

TEST_F(DenseToSparseExceptionTest, CooNnzMismatchDoesNotWriteOutput) {
  const auto p = MakeStatelessParam("COO");
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(*handle_, env_->stream(), p, host);
  std::vector<uint8_t> indexSentinel;
  std::vector<uint8_t> valueSentinel;
  ASSERT_TRUE(PrepareNnzMismatchRun(&ctx, &indexSentinel, &valueSentinel));
  auto rows = DeviceBuffer::copyFrom(indexSentinel.data(), indexSentinel.size());
  auto cols = DeviceBuffer::copyFrom(indexSentinel.data(), indexSentinel.size());
  auto values =
      DeviceBuffer::copyFrom(valueSentinel.data(), valueSentinel.size());
  ASSERT_EQ(aclsparseCooSetPointers(ctx.sparse.get(), rows.get(), cols.get(),
                                   values.get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseDenseToSparseConvert(
                handle_->get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                ctx.convertWorkspace->get()),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  std::vector<uint8_t> actualRows(indexSentinel.size());
  std::vector<uint8_t> actualCols(indexSentinel.size());
  std::vector<uint8_t> actualValues(valueSentinel.size());
  rows.copyToHost(actualRows.data(), actualRows.size());
  cols.copyToHost(actualCols.data(), actualCols.size());
  values.copyToHost(actualValues.data(), actualValues.size());
  EXPECT_EQ(actualRows, indexSentinel);
  EXPECT_EQ(actualCols, indexSentinel);
  EXPECT_EQ(actualValues, valueSentinel);
}

TEST_F(DenseToSparseExceptionTest,
       BellDirectConvertUsesFixedStandardStorage) {
  const auto p = MakeStatelessParam("BELL");
  const auto host = MakeDenseToSparseInput(p);
  DenseToSparseRunContext ctx(*handle_, env_->stream(), p, host);
  ASSERT_TRUE(PrepareDenseToSparseRun(&ctx));
  ASSERT_EQ(aclsparseDenseToSparseConvert(
                handle_->get(), ctx.dense.get(), ctx.sparse.get(),
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto golden = DenseToSparseGolden(
      DenseToSparseFormatOf(p.format), p.m, p.n, p.ld, true, 0,
      sizeof(uint32_t), false, host.dense, p.block_size, p.ell_cols,
      host.ellPattern);
  std::vector<uint8_t> actualValues(golden.values.size());
  ctx.bellValues->copyToHost(actualValues.data(), actualValues.size());
  EXPECT_EQ(actualValues, golden.values);
  std::vector<uint8_t> actualPattern(ctx.patternBytes.size());
  ctx.pattern->copyToHost(actualPattern.data(), actualPattern.size());
  EXPECT_EQ(actualPattern, PackDenseToSparseIndices(
                               golden.ellPattern, ACL_SPARSE_INDEX_32I));
}

struct DenseToSparseDeviceHeader {
  int32_t status = -1;
  uint64_t nnz = std::numeric_limits<uint64_t>::max();
};

DenseToSparseTilingData MakeSingleElementWhiteboxTiling(uint32_t base = 0) {
  DenseToSparseTilingData tiling{};
  tiling.rows = 1;
  tiling.cols = 1;
  tiling.ld = 1;
  tiling.unitCount = 1;
  // One unit covering the single column; the host normally derives this from
  // the matrix shape.
  tiling.majorChunk = 1;
  tiling.statusOffset = 0;
  tiling.nnzOffset = 32;
  tiling.level0Offset = 64;
  tiling.format = static_cast<uint32_t>(ACL_SPARSE_FORMAT_CSR);
  tiling.order = static_cast<uint32_t>(ACL_SPARSE_ORDER_ROW);
  tiling.base = base;
  tiling.offsetType = static_cast<uint32_t>(ACL_SPARSE_INDEX_32I);
  tiling.indexType = static_cast<uint32_t>(ACL_SPARSE_INDEX_32I);
  tiling.elementBytes = sizeof(uint32_t);
  tiling.numBlocks = 1;
  return tiling;
}

DenseToSparseDeviceHeader
ReadDenseToSparseHeader(const DeviceBuffer &workspace) {
  std::vector<uint8_t> header(64, 0xA5);
  workspace.copyToHost(header.data(), header.size());
  DenseToSparseDeviceHeader result;
  std::copy_n(header.data(), sizeof(result.status),
              reinterpret_cast<uint8_t *>(&result.status));
  std::copy_n(header.data() + 32, sizeof(result.nnz),
              reinterpret_cast<uint8_t *>(&result.nnz));
  return result;
}

TEST_F(DenseToSparseExceptionTest, DeviceStatusAndNnzUseIndependentHeaders) {
  const uint32_t input = 0x3f800000u;
  auto dense = DeviceBuffer::copyFrom(&input, sizeof(input));
  auto offsets = DeviceBuffer::alloc(2 * sizeof(int32_t));
  auto workspace = DeviceBuffer::alloc(96);
  const auto tiling = MakeSingleElementWhiteboxTiling();
  densetosparse_analysis_kernel_do(static_cast<uint8_t *>(dense.get()),
                                   static_cast<uint8_t *>(offsets.get()),
                                   static_cast<uint8_t *>(workspace.get()), 1,
                                   tiling, env_->stream());
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto header = ReadDenseToSparseHeader(workspace);
  EXPECT_EQ(header.status, 0);
  EXPECT_EQ(header.nnz, 1u);
}

TEST_F(DenseToSparseExceptionTest, I32BaseOverflowSetsDeviceStatus) {
  const uint32_t input = 0x3f800000u;
  auto dense = DeviceBuffer::copyFrom(&input, sizeof(input));
  auto offsets = DeviceBuffer::alloc(2 * sizeof(int32_t));
  auto workspace = DeviceBuffer::alloc(96);
  const auto tiling = MakeSingleElementWhiteboxTiling(
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  densetosparse_analysis_kernel_do(static_cast<uint8_t *>(dense.get()),
                                   static_cast<uint8_t *>(offsets.get()),
                                   static_cast<uint8_t *>(workspace.get()), 1,
                                   tiling, env_->stream());
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto header = ReadDenseToSparseHeader(workspace);
  EXPECT_NE(header.status, 0);
  EXPECT_EQ(header.nnz, 1u);
}

TEST_F(DenseToSparseExceptionTest, ConvertRebuildsNnzInDifferentWorkspace) {
  const uint32_t input = 0x3f800000u;
  auto dense = DeviceBuffer::copyFrom(&input, sizeof(input));
  auto offsets = DeviceBuffer::alloc(2 * sizeof(int32_t));
  auto indices = DeviceBuffer::alloc(2 * sizeof(int32_t));
  auto values = DeviceBuffer::alloc(2 * sizeof(uint32_t));
  auto analysisWorkspace = DeviceBuffer::alloc(96);
  auto matchingWorkspace = DeviceBuffer::alloc(96);
  auto mismatchingWorkspace = DeviceBuffer::alloc(96);

  auto tiling = MakeSingleElementWhiteboxTiling();
  densetosparse_analysis_kernel_do(
      static_cast<uint8_t *>(dense.get()),
      static_cast<uint8_t *>(offsets.get()),
      static_cast<uint8_t *>(analysisWorkspace.get()), 1, tiling,
      env_->stream());
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  ASSERT_EQ(ReadDenseToSparseHeader(analysisWorkspace).nnz, 1u);

  tiling.nnz = 1;
  densetosparse_convert_kernel_do(
      static_cast<uint8_t *>(dense.get()),
      static_cast<uint8_t *>(matchingWorkspace.get()),
      static_cast<uint8_t *>(offsets.get()),
      static_cast<uint8_t *>(indices.get()), nullptr, nullptr,
      static_cast<uint8_t *>(values.get()), nullptr, 1, tiling, env_->stream());
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto matching = ReadDenseToSparseHeader(matchingWorkspace);
  EXPECT_EQ(matching.status, 0);
  EXPECT_EQ(matching.nnz, 1u);

  tiling.nnz = 2;
  densetosparse_convert_kernel_do(
      static_cast<uint8_t *>(dense.get()),
      static_cast<uint8_t *>(mismatchingWorkspace.get()),
      static_cast<uint8_t *>(offsets.get()),
      static_cast<uint8_t *>(indices.get()), nullptr, nullptr,
      static_cast<uint8_t *>(values.get()), nullptr, 1, tiling, env_->stream());
  ASSERT_EQ(aclrtSynchronizeStream(env_->stream()), ACL_SUCCESS);
  const auto mismatching = ReadDenseToSparseHeader(mismatchingWorkspace);
  EXPECT_NE(mismatching.status, 0);
  EXPECT_EQ(mismatching.nnz, 1u);
}

TEST_F(DenseToSparseExceptionTest, NullPublicArguments) {
  size_t size = 0;
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                nullptr, nullptr, nullptr, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                &size),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), nullptr, nullptr,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), nullptr, nullptr,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  EXPECT_EQ(aclsparseDenseToSparseAnalysis(handle_->get(), nullptr, nullptr,
                                           ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                           nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  EXPECT_EQ(aclsparseDenseToSparseConvert(handle_->get(), nullptr, nullptr,
                                          ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                          nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(DenseToSparseExceptionTest, ConvertWithoutAnalysisForZeroNnzSucceeds) {
  uint32_t denseValue = 0;
  int32_t offset[2] = {0, 0};
  auto dDense = DeviceBuffer::copyFrom(&denseValue, sizeof(denseValue));
  auto dOffset = DeviceBuffer::copyFrom(offset, sizeof(offset));
  aclsparseConstDnMatDescr_t dense = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense, 1, 1, 1, dDense.get(), ACL_FLOAT,
                                      ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseSpMatDescr_t sparse = nullptr;
  ASSERT_EQ(aclsparseCreateCsr(&sparse, 1, 1, 0, dOffset.get(), nullptr,
                               nullptr, ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                               ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
  size_t workspace = 0;
  ASSERT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &workspace),
            ACL_SPARSE_STATUS_SUCCESS);
  auto buffer = DeviceBuffer::alloc(workspace);
  EXPECT_EQ(aclsparseDenseToSparseConvert(handle_->get(), dense, sparse,
                                          ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                          buffer.get()),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);
}

TEST_F(DenseToSparseExceptionTest, WorkspaceRequiredButNullIsRejected) {
  uint32_t denseValue = 0;
  int32_t offsets[2] = {0, 0};
  auto dDense = DeviceBuffer::copyFrom(&denseValue, sizeof(denseValue));
  auto dOffsets = DeviceBuffer::copyFrom(offsets, sizeof(offsets));
  aclsparseConstDnMatDescr_t dense = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense, 1, 1, 1, dDense.get(), ACL_FLOAT,
                                      ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseSpMatDescr_t sparse = nullptr;
  ASSERT_EQ(aclsparseCreateCsr(&sparse, 1, 1, 0, dOffsets.get(), nullptr,
                               nullptr, ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                               ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
  size_t workspace = 0;
  ASSERT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &workspace),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_GT(workspace, 0u);
  EXPECT_EQ(aclsparseDenseToSparseAnalysis(handle_->get(), dense, sparse,
                                           ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                           nullptr),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);
}

TEST_F(DenseToSparseExceptionTest, InvalidAlgAndUnsupportedDtype) {
  uint32_t value = 0;
  int32_t offsets[2] = {0, 0};
  aclsparseConstDnMatDescr_t dense = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense, 1, 1, 1, &value, ACL_FLOAT,
                                      ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseSpMatDescr_t sparse = nullptr;
  ASSERT_EQ(aclsparseCreateCsr(&sparse, 1, 1, 0, offsets, nullptr, nullptr,
                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
  size_t size = 0;
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                static_cast<aclsparseDenseToSparseAlg_t>(99), &size),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);

  double fp64 = 1.0;
  int32_t fp64Offsets[2] = {0, 0};
  dense = nullptr;
  sparse = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense, 1, 1, 1, &fp64, ACL_DOUBLE,
                                      ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  ASSERT_EQ(aclsparseCreateCsr(&sparse, 1, 1, 0, fp64Offsets, nullptr, nullptr,
                               ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                               ACL_SPARSE_INDEX_BASE_ZERO, ACL_DOUBLE),
            ACL_SPARSE_STATUS_SUCCESS);
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size),
            ACL_SPARSE_STATUS_NOT_SUPPORTED);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);
}

struct DenseDescriptorValidationCase {
  int64_t rows, cols, ld;
  bool nullValues;
  aclsparseStatus_t expected;
};

TEST(DenseToSparseDescriptorTest, EmptyShapesPreserveValidation) {
  uint32_t value = 0;
  using Case = DenseDescriptorValidationCase;
  const Case cases[] = {
      {0, 4, 4, false, ACL_SPARSE_STATUS_SUCCESS},
      {3, 0, 3, false, ACL_SPARSE_STATUS_SUCCESS},
      {0, 0, 1, false, ACL_SPARSE_STATUS_SUCCESS},
      {-1, 0, 1, false, ACL_SPARSE_STATUS_INVALID_VALUE},
      {0, -1, 1, false, ACL_SPARSE_STATUS_INVALID_VALUE},
      {0, 0, 0, false, ACL_SPARSE_STATUS_INVALID_VALUE},
      {0, 0, -1, false, ACL_SPARSE_STATUS_INVALID_VALUE},
      {0, 0, 1, true, ACL_SPARSE_STATUS_INVALID_VALUE},
  };
  for (bool isConst : {false, true}) {
    for (auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
      const auto check = [&](const Case &c) {
        SCOPED_TRACE(testing::Message()
                     << "const=" << isConst << " order=" << order
                     << " shape=" << c.rows << "x" << c.cols
                     << " ld=" << c.ld << " nullValues=" << c.nullValues);
        aclsparseDnMatDescr_t mutableDense = nullptr;
        aclsparseConstDnMatDescr_t constDense = nullptr;
        auto *values = c.nullValues ? nullptr : &value;
        const auto status = isConst
            ? aclsparseCreateConstDnMat(&constDense, c.rows, c.cols, c.ld,
                                        values, ACL_FLOAT, order)
            : aclsparseCreateDnMat(&mutableDense, c.rows, c.cols, c.ld,
                                   values, ACL_FLOAT, order);
        EXPECT_EQ(status, c.expected);
        const auto dense = isConst ? constDense : mutableDense;
        if (status == ACL_SPARSE_STATUS_SUCCESS) {
          ASSERT_NE(dense, nullptr);
          EXPECT_EQ(dense->rows, c.rows);
          EXPECT_EQ(dense->cols, c.cols);
          EXPECT_EQ(dense->ld, c.ld);
          EXPECT_EQ(aclsparseDestroyDnMat(dense), ACL_SPARSE_STATUS_SUCCESS);
        } else {
          EXPECT_EQ(dense, nullptr);
        }
      };
      for (const auto &c : cases)
        check(c);
      // Even an empty matrix must satisfy the order-specific leading dimension.
      check(order == ACL_SPARSE_ORDER_ROW
                ? Case{0, 4, 3, false, ACL_SPARSE_STATUS_INVALID_VALUE}
                : Case{4, 0, 3, false, ACL_SPARSE_STATUS_INVALID_VALUE});
    }
  }
}

TEST_F(DenseToSparseExceptionTest, Int32MaxAcceptedAndPlusOneRejectedByApi) {
  uint32_t value = 0;
  aclsparseConstDnMatDescr_t dense = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense,
                                      std::numeric_limits<int32_t>::max(), 0,
                                      std::numeric_limits<int32_t>::max(),
                                      &value, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseSpMatDescr_t sparse = nullptr;
  ASSERT_EQ(aclsparseCreateCoo(&sparse, std::numeric_limits<int32_t>::max(), 0,
                               0, nullptr, nullptr, nullptr,
                               ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO,
                               ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
  size_t workspace = 1;
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &workspace),
            ACL_SPARSE_STATUS_SUCCESS);
  EXPECT_EQ(workspace, 0u);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);

  dense = nullptr;
  ASSERT_EQ(
      aclsparseCreateConstDnMat(
          &dense, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
          0, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
          &value, ACL_FLOAT, ACL_SPARSE_ORDER_ROW),
      ACL_SPARSE_STATUS_SUCCESS);
  sparse = nullptr;
  ASSERT_EQ(aclsparseCreateCoo(
                &sparse,
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                0, 0, nullptr, nullptr, nullptr, ACL_SPARSE_INDEX_64I,
                ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT),
            ACL_SPARSE_STATUS_SUCCESS);
  EXPECT_EQ(aclsparseDenseToSparseGetBufferSize(
                handle_->get(), dense, sparse,
                ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &workspace),
            ACL_SPARSE_STATUS_INVALID_VALUE);
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);
}

struct DenseToSparseL2Param {
  std::string caseName;
  std::string mutation;
  aclsparseStatus_t expected = ACL_SPARSE_STATUS_INVALID_VALUE;
};

aclsparseStatus_t ParseDenseToSparseStatus(const std::string &name) {
  return name == "NOT_SUPPORTED" ? ACL_SPARSE_STATUS_NOT_SUPPORTED
                                 : ACL_SPARSE_STATUS_INVALID_VALUE;
}

std::vector<DenseToSparseL2Param> LoadDenseToSparseL2Cases() {
  std::vector<DenseToSparseL2Param> result;
  for (const auto &row : ReadMap("densetosparse_l2_cases.csv")) {
    result.push_back(
        {parseString(row, "case_name"), parseString(row, "mutation"),
         ParseDenseToSparseStatus(parseString(row, "expected_status"))});
  }
  return result;
}

struct DescriptorPair {
  aclsparseConstDnMatDescr_t dense = nullptr;
  aclsparseSpMatDescr_t sparse = nullptr;

  DescriptorPair() = default;
  DescriptorPair(const DescriptorPair &) = delete;
  DescriptorPair &operator=(const DescriptorPair &) = delete;
  DescriptorPair(DescriptorPair &&other) noexcept
      : dense(other.dense), sparse(other.sparse) {
    other.dense = nullptr;
    other.sparse = nullptr;
  }
  DescriptorPair &operator=(DescriptorPair &&other) noexcept {
    if (this != &other) {
      if (sparse != nullptr)
        aclsparseDestroySpMat(sparse);
      if (dense != nullptr)
        aclsparseDestroyDnMat(dense);
      dense = other.dense;
      sparse = other.sparse;
      other.dense = nullptr;
      other.sparse = nullptr;
    }
    return *this;
  }

  ~DescriptorPair() {
    if (sparse != nullptr)
      aclsparseDestroySpMat(sparse);
    if (dense != nullptr)
      aclsparseDestroyDnMat(dense);
  }
};

DescriptorPair MakeHostDescriptorPair(int64_t denseRows = 2,
                                      int64_t sparseRows = 2,
                                      aclDataType denseType = ACL_FLOAT,
                                      aclDataType sparseType = ACL_FLOAT) {
  static uint32_t denseValue[4] = {1, 0, 0, 1};
  static int32_t offsets[3] = {0, 0, 0};
  DescriptorPair pair;
  if (aclsparseCreateConstDnMat(&pair.dense, denseRows, 2, 2, denseValue,
                                denseType, ACL_SPARSE_ORDER_ROW) !=
      ACL_SPARSE_STATUS_SUCCESS) {
    return pair;
  }
  aclsparseCreateCsr(&pair.sparse, sparseRows, 2, 0, offsets, nullptr, nullptr,
                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                     ACL_SPARSE_INDEX_BASE_ZERO, sparseType);
  return pair;
}

aclsparseStatus_t RunNullOrPairL2Case(const std::string &mutation,
                                      HandleManager &handle) {
  size_t size = 0;
  auto pair = mutation == "b_shape_m_plus_1" ? MakeHostDescriptorPair(2, 3)
              : mutation == "b_value_type_differs"
                  ? MakeHostDescriptorPair(2, 2, ACL_FLOAT, ACL_FLOAT16)
                  : MakeHostDescriptorPair();
  if (mutation == "null_handle") {
    return aclsparseDenseToSparseGetBufferSize(
        nullptr, pair.dense, pair.sparse, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &size);
  }
  if (mutation == "null_mat_a") {
    return aclsparseDenseToSparseGetBufferSize(
        handle.get(), nullptr, pair.sparse,
        ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
  }
  if (mutation == "null_mat_b") {
    return aclsparseDenseToSparseGetBufferSize(
        handle.get(), pair.dense, nullptr, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &size);
  }
  return aclsparseDenseToSparseGetBufferSize(
      handle.get(), pair.dense, pair.sparse,
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      mutation == "null_output" ? nullptr : &size);
}

aclsparseStatus_t RunStaticDescriptorL2Case(const std::string &mutation,
                                            HandleManager &handle) {
  if (mutation == "negative_m") {
    int32_t offset = 0;
    aclsparseSpMatDescr_t sparse = nullptr;
    return aclsparseCreateCsr(&sparse, -1, 2, 0, &offset, nullptr, nullptr,
                              ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                              ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  uint32_t value = 0;
  aclsparseConstDnMatDescr_t dense = nullptr;
  if (mutation == "row_ld_n_minus_1") {
    return aclsparseCreateConstDnMat(&dense, 2, 2, 1, &value, ACL_FLOAT,
                                     ACL_SPARSE_ORDER_ROW);
  }
  if (mutation == "ld_times_outer_overflow") {
    int32_t offsets[3] = {0, 0, 0};
    DescriptorPair pair;
    const auto denseStatus = aclsparseCreateConstDnMat(
        &pair.dense, 2, 1, std::numeric_limits<int64_t>::max(), &value,
        ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    if (denseStatus != ACL_SPARSE_STATUS_SUCCESS)
      return denseStatus;
    const auto sparseStatus = aclsparseCreateCsr(
        &pair.sparse, 2, 1, 0, offsets, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    if (sparseStatus != ACL_SPARSE_STATUS_SUCCESS)
      return sparseStatus;
    size_t size = 0;
    return aclsparseDenseToSparseGetBufferSize(
        handle.get(), pair.dense, pair.sparse,
        ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
  }
  size_t size = 0;
  auto pair = MakeHostDescriptorPair(
      2, 2, mutation == "fp64_value_type" ? ACL_DOUBLE : ACL_FLOAT,
      mutation == "fp64_value_type" ? ACL_DOUBLE : ACL_FLOAT);
  return aclsparseDenseToSparseGetBufferSize(
      handle.get(), pair.dense, pair.sparse,
      mutation == "alg_enum_out_of_range"
          ? static_cast<aclsparseDenseToSparseAlg_t>(99)
          : ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &size);
}

aclsparseStatus_t RunEnumL2Case(const std::string &mutation,
                                HandleManager &handle) {
  size_t size = 0;
  auto pair = MakeHostDescriptorPair();
  auto *dense = const_cast<aclsparseDnMatDescr *>(
      reinterpret_cast<const aclsparseDnMatDescr *>(pair.dense));
  auto *sparse = reinterpret_cast<aclsparseSpMatDescr *>(pair.sparse);
  if (mutation == "format_enum_out_of_range") {
    sparse->format = static_cast<aclsparseFormat_t>(99);
  } else if (mutation == "order_enum_out_of_range") {
    dense->order = static_cast<aclsparseOrder_t>(99);
  } else if (mutation == "base_enum_out_of_range") {
    sparse->baseType = static_cast<aclsparseIndexBase_t>(99);
  } else {
    sparse->IdxType = static_cast<aclsparseIndexType_t>(99);
  }
  return aclsparseDenseToSparseGetBufferSize(
      handle.get(), pair.dense, pair.sparse,
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
}

aclsparseStatus_t RunDelayedBindingL2Case(HandleManager &handle) {
  const uint32_t input[4] = {1, 0, 0, 1};
  auto dDense = DeviceBuffer::copyFrom(input, sizeof(input));
  aclsparseConstDnMatDescr_t dense = nullptr;
  aclsparseSpMatDescr_t sparse = nullptr;
  const auto createStatus = aclsparseCreateCsr(
      &sparse, 2, 2, 0, nullptr, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  EXPECT_EQ(createStatus, ACL_SPARSE_STATUS_SUCCESS)
      << "generic CSR descriptor must permit delayed pointer binding";
  if (createStatus != ACL_SPARSE_STATUS_SUCCESS)
    return createStatus;
  const auto denseStatus = aclsparseCreateConstDnMat(
      &dense, 2, 2, 2, dDense.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
  EXPECT_EQ(denseStatus, ACL_SPARSE_STATUS_SUCCESS);
  if (denseStatus != ACL_SPARSE_STATUS_SUCCESS) {
    aclsparseDestroySpMat(sparse);
    return denseStatus;
  }
  size_t localSize = 0;
  const auto executionStatus = aclsparseDenseToSparseGetBufferSize(
      handle.get(), dense, sparse, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &localSize);
  aclsparseDestroyDnMat(dense);
  aclsparseDestroySpMat(sparse);
  return executionStatus;
}

aclsparseStatus_t RunMixedCompressedIndexL2Case(const std::string &mutation,
                                                HandleManager &handle) {
  uint32_t denseValue[4] = {1, 0, 0, 1};
  int64_t offsets[3] = {0, 0, 0};
  DescriptorPair pair;
  const auto denseStatus = aclsparseCreateConstDnMat(
      &pair.dense, 2, 2, 2, denseValue, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
  EXPECT_EQ(denseStatus, ACL_SPARSE_STATUS_SUCCESS);
  if (denseStatus != ACL_SPARSE_STATUS_SUCCESS)
    return denseStatus;

  const bool offsetI32 = mutation.find("i32_i64") != std::string::npos;
  const auto offsetType =
      offsetI32 ? ACL_SPARSE_INDEX_32I : ACL_SPARSE_INDEX_64I;
  const auto indexType =
      offsetI32 ? ACL_SPARSE_INDEX_64I : ACL_SPARSE_INDEX_32I;
  aclsparseStatus_t sparseStatus = ACL_SPARSE_STATUS_INVALID_VALUE;
  if (mutation.rfind("csr_", 0) == 0) {
    sparseStatus = aclsparseCreateCsr(&pair.sparse, 2, 2, 0, offsets, nullptr,
                                      nullptr, offsetType, indexType,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  } else {
    sparseStatus = aclsparseCreateCsc(&pair.sparse, 2, 2, 0, offsets, nullptr,
                                      nullptr, offsetType, indexType,
                                      ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  EXPECT_EQ(sparseStatus, ACL_SPARSE_STATUS_SUCCESS)
      << "generic compressed descriptor must accept mixed index types";
  if (sparseStatus != ACL_SPARSE_STATUS_SUCCESS)
    return sparseStatus;

  size_t size = 0;
  return aclsparseDenseToSparseGetBufferSize(
      handle.get(), pair.dense, pair.sparse,
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
}

aclsparseStatus_t RunWorkspaceStateL2Case(const std::string &mutation,
                                          HandleManager &handle,
                                          aclrtStream stream) {
  size_t size = 0;
  const uint32_t input[4] = {1, 0, 0, 1};
  const int32_t offsets[3] = {0, 0, 0};
  auto dDense = DeviceBuffer::copyFrom(input, sizeof(input));
  auto dOffsets = DeviceBuffer::copyFrom(offsets, sizeof(offsets));
  DescriptorPair pair;
  aclsparseCreateConstDnMat(&pair.dense, 2, 2, 2, dDense.get(), ACL_FLOAT,
                            ACL_SPARSE_ORDER_ROW);
  aclsparseCreateCsr(&pair.sparse, 2, 2, 0, dOffsets.get(), nullptr, nullptr,
                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  handle.setStream(stream);
  aclsparseDenseToSparseGetBufferSize(handle.get(), pair.dense, pair.sparse,
                                      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                      &size);
  if (mutation == "workspace_gt_zero_null") {
    return aclsparseDenseToSparseAnalysis(handle.get(), pair.dense, pair.sparse,
                                          ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                          nullptr);
  }
  auto workspace = DeviceBuffer::alloc(size);
  const auto analysis = aclsparseDenseToSparseAnalysis(
      handle.get(), pair.dense, pair.sparse,
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, workspace.get());
  if (analysis != ACL_SPARSE_STATUS_SUCCESS)
    return analysis;
  return aclsparseDenseToSparseConvert(handle.get(), pair.dense, pair.sparse,
                                       ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                       workspace.get());
}

aclsparseStatus_t RunBellL2Case(const std::string &mutation) {
  uint32_t value = 0;
  int32_t pattern = 0;
  aclsparseSpMatDescr_t bell = nullptr;
  if (mutation == "bell_block_size_zero") {
    return aclsparseCreateBlockedEll(&bell, 4, 4, 0, 2, &pattern, &value,
                                     ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  if (mutation == "bell_tail_values_null") {
    // rows=5 leaves a tail block: capacity is the padded 3x1 block grid, so
    // ellValue is still required even though rows % ellBlockSize != 0.
    return aclsparseCreateBlockedEll(&bell, 5, 4, 2, 2, &pattern, nullptr,
                                     ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  if (mutation == "bell_ellcols_n_plus_block") {
    return aclsparseCreateBlockedEll(&bell, 4, 4, 2, 6, &pattern, &value,
                                     ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  if (mutation == "bell_nonzero_pattern_null") {
    return aclsparseCreateBlockedEll(&bell, 4, 4, 2, 2, nullptr, &value,
                                     ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  if (mutation == "bell_nonzero_values_null") {
    return aclsparseCreateBlockedEll(&bell, 4, 4, 2, 2, &pattern, nullptr,
                                     ACL_SPARSE_INDEX_32I,
                                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  if (mutation == "bell_ceildiv_overflow") {
    // ellCols 接近 INT64_MAX：CeilDiv(ellCols, ellBlockSize) 的旧写法
    // (value + divisor - 1) 会有符号溢出，且发生在容量检查之前。正确实现应在
    // slots 超出补齐后的块列数时返回 INVALID_VALUE，而非触发 UB。
    return aclsparseCreateBlockedEll(
        &bell, 0, std::numeric_limits<int64_t>::max() - 2, 2,
        std::numeric_limits<int64_t>::max(), &pattern, &value,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  }
  return aclsparseCreateBlockedEll(
      &bell, std::numeric_limits<int64_t>::max() - 1,
      std::numeric_limits<int64_t>::max() - 1, 1, 2, &pattern, &value,
      ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
}

aclsparseStatus_t RunTheoreticalL2Case(const std::string &mutation,
                                       HandleManager &handle) {
  if (mutation == "nnz_plus_base_int32_over") {
    const uint64_t nnz = std::numeric_limits<int32_t>::max();
    const uint64_t base = 1;
    return nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) -
                       base
               ? ACL_SPARSE_STATUS_INVALID_VALUE
               : ACL_SPARSE_STATUS_SUCCESS;
  }
  const int64_t over =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  auto pair = MakeHostDescriptorPair();
  auto *sparse = reinterpret_cast<aclsparseSpMatDescr *>(pair.sparse);
  if (mutation == "m_int32_max_plus_1")
    sparse->rows = over;
  else
    sparse->cols = over;
  size_t size = 0;
  return aclsparseDenseToSparseGetBufferSize(
      handle.get(), pair.dense, pair.sparse,
      ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
}

aclsparseStatus_t RunNonzeroDenseNullL2Case(HandleManager &handle) {
  aclsparseConstDnMatDescr_t dense = nullptr;
  int32_t offsets[3] = {0, 0, 0};
  aclsparseSpMatDescr_t sparse = nullptr;
  aclsparseCreateConstDnMat(&dense, 2, 2, 2, nullptr, ACL_FLOAT,
                            ACL_SPARSE_ORDER_ROW);
  aclsparseCreateCsr(&sparse, 2, 2, 0, offsets, nullptr, nullptr,
                     ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                     ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
  size_t size = 0;
  const auto status = aclsparseDenseToSparseGetBufferSize(
      handle.get(), dense, sparse, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT, &size);
  if (sparse)
    aclsparseDestroySpMat(sparse);
  if (dense)
    aclsparseDestroyDnMat(dense);
  return status;
}

aclsparseStatus_t RunDenseToSparseL2Case(const DenseToSparseL2Param &p,
                                         HandleManager &handle,
                                         aclrtStream stream) {
  using Handler = aclsparseStatus_t (*)(const std::string &, HandleManager &);
  static const std::vector<std::pair<std::vector<std::string>, Handler>>
      groups = {
          {{"null_handle", "null_mat_a", "null_mat_b", "null_output",
            "b_shape_m_plus_1", "b_value_type_differs"},
           RunNullOrPairL2Case},
          {{"negative_m", "row_ld_n_minus_1", "ld_times_outer_overflow",
            "alg_enum_out_of_range", "fp64_value_type"},
           RunStaticDescriptorL2Case},
          {{"format_enum_out_of_range", "order_enum_out_of_range",
            "base_enum_out_of_range", "index_enum_out_of_range"},
           RunEnumL2Case},
          {{"m_int32_max_plus_1", "n_int32_max_plus_1",
            "nnz_plus_base_int32_over"},
           RunTheoreticalL2Case},
      };
  for (const auto &group : groups) {
    if (std::find(group.first.begin(), group.first.end(), p.mutation) !=
        group.first.end())
      return group.second(p.mutation, handle);
  }
  if (p.mutation == "nonzero_dense_null")
    return RunNonzeroDenseNullL2Case(handle);
  if (p.mutation == "nonzero_logical_array_null") {
    return RunDelayedBindingL2Case(handle);
  }
  if (p.mutation.rfind("csr_mixed_", 0) == 0 ||
      p.mutation.rfind("csc_mixed_", 0) == 0) {
    return RunMixedCompressedIndexL2Case(p.mutation, handle);
  }
  if (p.mutation.rfind("bell_", 0) == 0)
    return RunBellL2Case(p.mutation);
  return RunWorkspaceStateL2Case(p.mutation, handle, stream);
}

using DenseToSparseL2Test = DenseToSparseParamTest<DenseToSparseL2Param>;

TEST_P(DenseToSparseL2Test, ReturnsExpectedStatus) {
  const auto &p = GetParam();
  EXPECT_EQ(RunDenseToSparseL2Case(p, *handle_, env_->stream()), p.expected)
      << p.caseName << " mutation=" << p.mutation;
}

INSTANTIATE_TEST_SUITE_P(
    CsvL2, DenseToSparseL2Test, testing::ValuesIn(LoadDenseToSparseL2Cases()),
    [](const testing::TestParamInfo<DenseToSparseL2Param> &info) {
      return info.param.caseName;
    });

std::vector<DenseToSparseParam> LoadDenseToSparseBoundaryCases() {
  auto cases = GetCasesFromCsv<DenseToSparseParam>("densetosparse_test.csv");
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [](const DenseToSparseParam &p) {
                               return !p.descriptorOnly() &&
                                      !p.bufferSizeOnly();
                             }),
              cases.end());
  return cases;
}

using DenseToSparseBoundaryTest = DenseToSparseParamTest<DenseToSparseParam>;

TEST_P(DenseToSparseBoundaryTest, DescriptorOrBufferSizeOnly) {
  const auto &p = GetParam();
  static uint32_t denseValue = 0;
  static int64_t offsets[2] = {0, 0};
  const bool reject = p.m > std::numeric_limits<int32_t>::max() ||
                      p.n > std::numeric_limits<int32_t>::max();
  aclsparseConstDnMatDescr_t dense = nullptr;
  ASSERT_EQ(aclsparseCreateConstDnMat(&dense, p.m, p.n, p.ld, &denseValue,
                                      ACL_FLOAT, DenseToSparseOrder(p.order)),
            ACL_SPARSE_STATUS_SUCCESS);
  aclsparseSpMatDescr_t sparse = nullptr;
  const auto createSparse = p.format == "CSC" ? aclsparseCreateCsc
                                             : aclsparseCreateCsr;
  const auto createStatus =
      createSparse(&sparse, p.m, p.n, 0, offsets, nullptr, nullptr,
                         DenseToSparseIndexType(p.offset_type),
                         DenseToSparseIndexType(p.index_type),
                         DenseToSparseBase(p.base), ACL_FLOAT);
  ASSERT_EQ(createStatus, ACL_SPARSE_STATUS_SUCCESS);
  size_t size = 0;
  const auto status = aclsparseDenseToSparseGetBufferSize(
      handle_->get(), dense, sparse, ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &size);
  EXPECT_EQ(status, reject ? ACL_SPARSE_STATUS_INVALID_VALUE
                           : ACL_SPARSE_STATUS_SUCCESS)
      << p.caseId();
  aclsparseDestroySpMat(sparse);
  aclsparseDestroyDnMat(dense);
}

INSTANTIATE_TEST_SUITE_P(
    CsvTheoreticalBoundary, DenseToSparseBoundaryTest,
    testing::ValuesIn(LoadDenseToSparseBoundaryCases()),
    [](const testing::TestParamInfo<DenseToSparseParam> &info) {
      return info.param.caseId();
    });

} // namespace
