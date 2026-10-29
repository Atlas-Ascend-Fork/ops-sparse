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
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aclsparse_descr_internal.h"
#include "sparse2dense_golden.h"
#include "sparse2dense_npu_wrapper.h"
#include "sparse2dense_param.h"
#include "sparse_test.h"

using namespace sparse_test;

namespace {

template <typename Base>
class Sparse2DenseAclEnvironment : public Base {
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
class Sparse2DenseParamTest
    : public Sparse2DenseAclEnvironment<testing::TestWithParam<Param>> {};

using Sparse2DenseTest = Sparse2DenseParamTest<Sparse2DenseParam>;

void ExpectSparse2DenseStatuses(const Sparse2DenseParam &p,
                                 const Sparse2DenseRunResult &actual) {
    ASSERT_EQ(actual.descriptorStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
    ASSERT_EQ(actual.queryStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
    ASSERT_EQ(actual.executeStatus, ACL_SPARSE_STATUS_SUCCESS) << p.caseId();
    ASSERT_EQ(actual.syncStatus, ACL_SUCCESS) << p.caseId();
    EXPECT_EQ(actual.workspaceSize, 0u) << p.caseId();
}

void ExpectSparse2DenseGolden(const Sparse2DenseParam &p,
                               const Sparse2DenseRunResult &actual) {
    const auto golden = Sparse2DenseGolden(
        p.format, p.m, p.n, p.ld, p.order == "ROW",
        p.base == "ONE" ? 1 : 0,
        Sparse2DenseValueType(p.value_type),
        p.distribution, p.seed);
    if (actual.dense != golden.dense) {
        size_t mismatch = 0;
        const size_t n = std::min(actual.dense.size(), golden.dense.size());
        while (mismatch < n && actual.dense[mismatch] == golden.dense[mismatch]) {
            ++mismatch;
        }
        ADD_FAILURE() << p.caseId()
                      << " dense mismatch at byte " << mismatch
                      << " actual_size=" << actual.dense.size()
                      << " golden_size=" << golden.dense.size()
                      << (mismatch < n
                              ? (std::string(" actual_byte=") +
                                 std::to_string(actual.dense[mismatch]) +
                                 " golden_byte=" +
                                 std::to_string(golden.dense[mismatch]))
                              : std::string(" (size mismatch)"));
    }
}

TEST_P(Sparse2DenseTest, BitwiseGolden) {
    const auto &p = GetParam();
    const auto host = MakeSparse2DenseInput(p);
    const auto actual = RunSparse2Dense(*handle_, env_->stream(), p, host);
    // Empty shape (m==0 or n==0): still require SUCCESS + stream sync + all-zero
    // physical plane (majorDim * ld * esize, may be 0 bytes).
    ExpectSparse2DenseStatuses(p, actual);
    ExpectSparse2DenseGolden(p, actual);
}

std::vector<Sparse2DenseParam> LoadSparse2DenseCases() {
    return GetCasesFromCsv<Sparse2DenseParam>("sparse2dense_test.csv");
}

INSTANTIATE_TEST_SUITE_P(
    CsvL0L1, Sparse2DenseTest,
    testing::ValuesIn(LoadSparse2DenseCases()),
    [](const testing::TestParamInfo<Sparse2DenseParam> &info) {
        std::string name = info.param.caseId();
        std::replace_if(
            name.begin(), name.end(),
            [](char c) {
                return !(std::isalnum(static_cast<unsigned char>(c)) || c == '_');
            },
            '_');
        return name;
    });

static aclsparseStatus_t InvokeMutation(const std::string &mutation,
    aclsparseHandle_t handle, aclsparseSpMatDescr_t sparse,
    aclsparseDnMatDescr_t dense, aclsparseSpMatDescr *sparseInner,
    aclsparseSparseToDenseAlg_t alg, size_t &size)
{
    if (mutation == "null_handle")
        return aclsparseSparseToDense_bufferSize(nullptr, sparse, dense, alg, &size);
    if (mutation == "null_mat_a")
        return aclsparseSparseToDense_bufferSize(handle, nullptr, dense, alg, &size);
    if (mutation == "null_mat_b")
        return aclsparseSparseToDense_bufferSize(handle, sparse, nullptr, alg, &size);
    if (mutation == "null_output")
        return aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, nullptr);
    if (mutation == "nonzero_nnz_ptrs_null") {
        sparseInner->nnz = 4;
        sparseInner->ptrs = nullptr;
        return aclsparseSparseToDense(handle, sparse, dense, alg, nullptr);
    }
    if (mutation == "nonzero_nnz_idxs_null") {
        sparseInner->nnz = 4;
        sparseInner->idxs = nullptr;
        return aclsparseSparseToDense(handle, sparse, dense, alg, nullptr);
    }
    // dense->values==nullptr: bufferSize must still succeed; execute rejects.
    if (mutation == "nonzero_dense_null") {
        size_t probe = 0;
        auto qs = aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, &probe);
        if (qs != ACL_SPARSE_STATUS_SUCCESS) {
            return qs;
        }
        size = probe;
        return aclsparseSparseToDense(handle, sparse, dense, alg, nullptr);
    }
    if (mutation == "nonzero_nnz_values_null") {
        sparseInner->nnz = 4;
        sparseInner->values = nullptr;
        return aclsparseSparseToDense(handle, sparse, dense, alg, nullptr);
    }
    if (mutation == "nonzero_nnz_ptrs_null_buffer_size") {
        sparseInner->nnz = 4;
        sparseInner->ptrs = nullptr;
        return aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, &size);
    }
    if (mutation == "nonzero_nnz_idxs_null_buffer_size") {
        sparseInner->nnz = 4;
        sparseInner->idxs = nullptr;
        return aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, &size);
    }
    if (mutation == "nonzero_nnz_values_null_buffer_size") {
        sparseInner->nnz = 4;
        sparseInner->values = nullptr;
        return aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, &size);
    }
    return aclsparseSparseToDense_bufferSize(handle, sparse, dense, alg, &size);
}

static aclsparseStatus_t CreateL2Descriptors(const std::string &mutation,
    void *dDense, void *dOffsets, void *dColInd, void *dValues,
    aclsparseDnMatDescr_t &dense, aclsparseSpMatDescr_t &sparse,
    aclsparseSpMatDescr *&sparseInner)
{
    auto denseType = mutation == "b_value_type_differs" ? ACL_FLOAT16 : ACL_FLOAT;
    auto sparseType = mutation == "fp64_value_type" ? ACL_DOUBLE
                     : mutation == "complex64_value_type" ? ACL_COMPLEX64
                     : mutation == "b_value_type_differs" ? ACL_FLOAT : ACL_FLOAT;

    int64_t m = 2, n = 2;
    int64_t sparseM = mutation == "b_shape_m_plus_1" ? 3 : m;
    int64_t ld = mutation == "row_ld_n_minus_1" ? 1 : n;

    aclsparseStatus_t createSt = aclsparseCreateDnMat(&dense, m, n, ld, dDense, denseType,
                          ACL_SPARSE_ORDER_ROW);
    if (createSt != ACL_SPARSE_STATUS_SUCCESS) return createSt;

    auto fmt = ACL_SPARSE_FORMAT_CSR;
    if (mutation == "format_enum_out_of_range")
        fmt = static_cast<aclsparseFormat_t>(99);

    createSt = aclsparseCreateCsr(&sparse, sparseM, n, 0, dOffsets, dColInd,
                        dValues, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                        Sparse2DenseBase(mutation == "base_enum_out_of_range" ? "" : "ZERO"),
                        sparseType);
    if (createSt != ACL_SPARSE_STATUS_SUCCESS) return createSt;

    sparseInner = reinterpret_cast<aclsparseSpMatDescr *>(sparse);
    sparseInner->format = fmt;
    if (mutation == "i64_offset_type") {
        sparseInner->ptrType = ACL_SPARSE_INDEX_64I;
        sparseInner->IdxType = ACL_SPARSE_INDEX_64I;
    }

    if (mutation == "nonzero_dense_null") {
        aclsparseDestroyDnMat(dense);
        dense = nullptr;
        createSt = aclsparseCreateDnMat(&dense, 2, 2, 2, nullptr, ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW);
        if (createSt != ACL_SPARSE_STATUS_SUCCESS) return createSt;
    }
    return ACL_SPARSE_STATUS_SUCCESS;
}

aclsparseStatus_t RunL2Case(const std::string &mutation, HandleManager &handle) {
    size_t size = 0;
    const uint32_t denseValue[4] = {1, 0, 0, 1};
    const int32_t offsets[5] = {0, 0, 0, 0, 0};
    const int32_t colInd[4] = {0, 0, 0, 0};
    const uint32_t values[4] = {1, 1, 1, 1};

    auto dDense = DeviceBuffer::copyFrom(denseValue, sizeof(denseValue));
    auto dOffsets = DeviceBuffer::copyFrom(offsets, sizeof(offsets));
    auto dColInd = DeviceBuffer::copyFrom(colInd, sizeof(colInd));
    auto dValues = DeviceBuffer::copyFrom(values, sizeof(values));

    aclsparseDnMatDescr_t dense = nullptr;
    aclsparseSpMatDescr_t sparse = nullptr;

    if (mutation == "m_int32_max_plus_1") {
        int64_t bigM = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
        aclsparseCreateDnMat(&dense, bigM, 1, bigM, dDense.get(), ACL_FLOAT,
                              ACL_SPARSE_ORDER_ROW);
        aclsparseCreateCsr(&sparse, bigM, 1, 0, dOffsets.get(), nullptr, nullptr,
                            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
                            ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
        auto st = aclsparseSparseToDense_bufferSize(
            handle.get(), sparse, dense,
            ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &size);
        aclsparseDestroySpMat(sparse);
        aclsparseDestroyDnMat(dense);
        return st;
    }

    aclsparseSpMatDescr *sparseInner = nullptr;
    aclsparseStatus_t createSt = CreateL2Descriptors(mutation,
        dDense.get(), dOffsets.get(), dColInd.get(), dValues.get(),
        dense, sparse, sparseInner);
    if (createSt != ACL_SPARSE_STATUS_SUCCESS) {
        aclsparseDestroyDnMat(dense);
        aclsparseDestroySpMat(sparse);
        return createSt;
    }

    auto alg = mutation == "alg_enum_out_of_range"
                   ? static_cast<aclsparseSparseToDenseAlg_t>(99)
                   : ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT;

    aclsparseStatus_t st = InvokeMutation(mutation, handle.get(), sparse, dense,
                                          sparseInner, alg, size);
    aclsparseDestroySpMat(sparse);
    aclsparseDestroyDnMat(dense);
    return st;
}

aclsparseStatus_t ParseExpectedStatus(const std::string &name) {
    if (name == "HANDLE_IS_NULLPTR") return ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR;
    if (name == "NOT_SUPPORTED")     return ACL_SPARSE_STATUS_NOT_SUPPORTED;
    return ACL_SPARSE_STATUS_INVALID_VALUE;
}

struct Sparse2DenseL2Param {
    std::string caseName;
    std::string mutation;
    aclsparseStatus_t expected = ACL_SPARSE_STATUS_INVALID_VALUE;
};

std::vector<Sparse2DenseL2Param> LoadSparse2DenseL2Cases() {
    std::vector<Sparse2DenseL2Param> result;
    for (const auto &row : ReadMap("sparse2dense_l2_cases.csv")) {
        result.push_back(
            {parseString(row, "case_name"),
             parseString(row, "mutation"),
             ParseExpectedStatus(parseString(row, "expected_status"))});
    }
    return result;
}

using Sparse2DenseL2Test =
    Sparse2DenseAclEnvironment<testing::TestWithParam<Sparse2DenseL2Param>>;

TEST_P(Sparse2DenseL2Test, NegativeCase) {
    const auto &p = GetParam();
    const auto actual = RunL2Case(p.mutation, *handle_);
    EXPECT_EQ(actual, p.expected) << p.caseName;
}


using Sparse2DenseUnboundPointerTest = Sparse2DenseAclEnvironment<testing::Test>;

TEST_F(Sparse2DenseUnboundPointerTest, RejectsUnboundPointersInBothStages) {
    auto dDense = DeviceBuffer::alloc(4 * sizeof(float));
    const std::array<const char *, 3> formats = {"CSR", "CSC", "COO"};

    for (const char *format : formats) {
        SpMatManager sparse;
        if (std::string(format) == "CSR") {
            sparse = SpMatManager::createCsr(2, 2, 1, nullptr, nullptr, nullptr);
        } else if (std::string(format) == "CSC") {
            sparse = SpMatManager::createCsc(2, 2, 1, nullptr, nullptr, nullptr);
        } else {
            sparse = SpMatManager::createCoo(2, 2, 1, nullptr, nullptr, nullptr);
        }
        auto dense = DnMatManager::create(2, 2, 2, dDense.get(), ACL_FLOAT,
                                          ACL_SPARSE_ORDER_ROW);

        size_t workspaceSize = 0;
        EXPECT_EQ(aclsparseSparseToDense_bufferSize(
                      handle_->get(), sparse.get(), dense.get(),
                      ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &workspaceSize),
                  ACL_SPARSE_STATUS_INVALID_VALUE)
            << format;
        EXPECT_EQ(aclsparseSparseToDense(
                      handle_->get(), sparse.get(), dense.get(),
                      ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr),
                  ACL_SPARSE_STATUS_INVALID_VALUE)
            << format;
    }
}

TEST_F(Sparse2DenseUnboundPointerTest, AllowsNullSparsePointersWhenNnzIsZero) {
    auto dDense = DeviceBuffer::alloc(4 * sizeof(float));
    const std::array<const char *, 3> formats = {"CSR", "CSC", "COO"};

    for (const char *format : formats) {
        SpMatManager sparse;
        if (std::string(format) == "CSR") {
            sparse = SpMatManager::createCsr(2, 2, 0, nullptr, nullptr, nullptr);
        } else if (std::string(format) == "CSC") {
            sparse = SpMatManager::createCsc(2, 2, 0, nullptr, nullptr, nullptr);
        } else {
            sparse = SpMatManager::createCoo(2, 2, 0, nullptr, nullptr, nullptr);
        }
        auto dense = DnMatManager::create(2, 2, 2, dDense.get(), ACL_FLOAT,
                                          ACL_SPARSE_ORDER_ROW);

        size_t workspaceSize = 0;
        EXPECT_EQ(aclsparseSparseToDense_bufferSize(
                      handle_->get(), sparse.get(), dense.get(),
                      ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, &workspaceSize),
                  ACL_SPARSE_STATUS_SUCCESS)
            << format;
        EXPECT_EQ(aclsparseSparseToDense(
                      handle_->get(), sparse.get(), dense.get(),
                      ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr),
                  ACL_SPARSE_STATUS_SUCCESS)
            << format;
    }
}

INSTANTIATE_TEST_SUITE_P(
    CsvL2, Sparse2DenseL2Test,
    testing::ValuesIn(LoadSparse2DenseL2Cases()),
    [](const testing::TestParamInfo<Sparse2DenseL2Param> &info) {
        return info.param.caseName;
    });

} // namespace