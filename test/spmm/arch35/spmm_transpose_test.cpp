/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <complex>
#include <tuple>
#include <vector>

#include "spmm_test_utils.h"

extern "C" aclsparseStatus_t aclsparseSpMMPrivateOutOfPlace(
    aclsparseHandle_t, aclsparseOperation_t, aclsparseOperation_t, const void *,
    aclsparseConstSpMatDescr_t, aclsparseConstDnMatDescr_t, const void *,
    aclsparseConstDnMatDescr_t, aclsparseDnMatDescr_t, aclDataType,
    aclsparseSpMMAlg_t, void *);

namespace {

using Complex = std::complex<float>;
using TransposeParam = std::tuple<aclDataType, aclsparseOperation_t, int32_t>;
constexpr int64_t kRows = 3;
constexpr int64_t kColumns = 5;
constexpr int64_t kN = 4;
constexpr int64_t kNnz = 6;

size_t ValueSize(aclDataType dtype)
{
    return dtype == ACL_COMPLEX64 ? sizeof(Complex) :
        (dtype == ACL_FLOAT || dtype == ACL_INT32 ? sizeof(float) :
        (dtype == ACL_INT8 ? 1 : sizeof(uint16_t)));
}

std::vector<uint8_t> Encode(const std::vector<Complex> &values, aclDataType dtype)
{
    std::vector<uint8_t> result(values.size() * ValueSize(dtype));
    for (size_t index = 0; index < values.size(); ++index) {
        const float real = values[index].real();
        auto *destination = result.data() + index * ValueSize(dtype);
        if (dtype == ACL_COMPLEX64) {
            const float parts[] = {real, values[index].imag()};
            spmm_test::SecureCopy(destination, sizeof(parts), parts, sizeof(parts));
        } else if (dtype == ACL_FLOAT) {
            spmm_test::SecureCopy(destination, sizeof(real), &real, sizeof(real));
        } else if (dtype == ACL_INT8) {
            const int8_t value = static_cast<int8_t>(real);
            spmm_test::SecureCopy(destination, sizeof(value), &value, sizeof(value));
        } else if (dtype == ACL_INT32) {
            const int32_t value = static_cast<int32_t>(real);
            spmm_test::SecureCopy(destination, sizeof(value), &value, sizeof(value));
        } else {
            const uint16_t value = dtype == ACL_FLOAT16 ?
                spmm_test::FloatToHalf(real) : spmm_test::FloatToBf16(real);
            spmm_test::SecureCopy(destination, sizeof(value), &value, sizeof(value));
        }
    }
    return result;
}

bool Upload(void *destination, const std::vector<Complex> &values, aclDataType dtype)
{
    const auto bytes = Encode(values, dtype);
    if (bytes.empty()) return true;
    return aclrtMemcpy(destination, bytes.size(), bytes.data(), bytes.size(),
        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

uint64_t DenseIndex(int64_t row, int64_t column, int64_t ld, aclsparseOrder_t order)
{
    return order == ACL_SPARSE_ORDER_ROW ? row * ld + column : column * ld + row;
}

struct RunOptions {
    aclsparseOperation_t opB = ACL_SPARSE_OP_NON_TRANSPOSE;
    aclsparseOrder_t bOrder = ACL_SPARSE_ORDER_ROW;
    aclsparseOrder_t cOrder = ACL_SPARSE_ORDER_ROW;
    bool preprocess = false;
    bool deviceScalars = false;
    bool separateOutput = false;
    bool mixedOutput = false;
    bool square = false;
    bool empty = false;
    aclsparseSpMMAlg_t algorithm = ACL_SPARSE_SPMM_CSR_ALG2;
};

class SpmmTransposeTest : public spmm_test::SpmmDeviceTest,
                          public ::testing::WithParamInterface<TransposeParam> {
protected:
    void TearDown() override
    {
        for (auto descriptor : sparse_) aclsparseDestroySpMat(descriptor);
        for (auto descriptor : dense_) aclsparseDestroyDnMat(descriptor);
        if (handle_ != nullptr) aclsparseDestroy(handle_);
        for (void *buffer : buffers_) aclrtFree(buffer);
        spmm_test::SpmmDeviceTest::TearDown();
    }

    void *Allocate(size_t bytes)
    {
        void *buffer = nullptr;
        if (aclrtMalloc(&buffer, std::max<size_t>(bytes, 1),
            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
        buffers_.push_back(buffer);
        return buffer;
    }

    bool CreateSparse(const std::vector<int32_t> &columns, void *values,
                      aclsparseSpMatDescr_t *descriptor)
    {
        const int32_t base = std::get<2>(GetParam());
        std::vector<int32_t> rows = options_.empty ?
            std::vector<int32_t>(kRows + 1, 0) : std::vector<int32_t>{0, 2, 4, 6};
        auto indices = columns;
        for (auto &value : rows) value += base;
        for (auto &value : indices) value += base;
        void *dRows = Allocate(rows.size() * sizeof(int32_t));
        void *dColumns = Allocate(indices.size() * sizeof(int32_t));
        if (dRows == nullptr || dColumns == nullptr) return false;
        if (aclrtMemcpy(dRows, rows.size() * sizeof(int32_t), rows.data(),
            rows.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(dColumns, indices.size() * sizeof(int32_t), indices.data(),
            indices.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return false;
        if (aclsparseCreateCsr(descriptor, kRows, columnCount_, options_.empty ? 0 : kNnz, dRows, dColumns,
            values, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE,
            std::get<0>(GetParam())) != ACL_SPARSE_STATUS_SUCCESS) return false;
        sparse_.push_back(*descriptor);
        return true;
    }

    bool CreateDense(int64_t rows, int64_t columns, int64_t ld,
                     aclsparseOrder_t order, aclDataType dtype,
                     aclsparseDnMatDescr_t *descriptor, void **values)
    {
        *values = Allocate((order == ACL_SPARSE_ORDER_ROW ? rows : columns) * ld * ValueSize(dtype));
        if (*values == nullptr || aclsparseCreateDnMat(descriptor, rows, columns, ld,
            *values, dtype, order) != ACL_SPARSE_STATUS_SUCCESS) return false;
        dense_.push_back(*descriptor);
        return true;
    }

    std::vector<Complex> Values(int offset) const
    {
        std::vector<Complex> values(options_.empty ? 0 : kNnz);
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = Complex(static_cast<float>((index + offset) % 5) - 2,
                std::get<0>(GetParam()) == ACL_COMPLEX64 ? static_cast<float>(index % 3) - 1 : 0);
        }
        return values;
    }

    void CheckResult(void *output, const std::vector<Complex> &expected,
                     aclDataType dtype)
    {
        const auto bytes = Encode(expected, dtype);
        std::vector<uint8_t> actual(bytes.size());
        ASSERT_EQ(aclrtMemcpy(actual.data(), actual.size(), output, actual.size(),
            ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        for (size_t index = 0; index < expected.size(); ++index) {
            const uint8_t *element = actual.data() + index * ValueSize(dtype);
            if (dtype == ACL_COMPLEX64 || dtype == ACL_FLOAT) {
                float parts[2] = {0, 0};
                ASSERT_TRUE(spmm_test::SecureCopy(parts, sizeof(parts), element, ValueSize(dtype)));
                EXPECT_FLOAT_EQ(parts[0], expected[index].real());
                if (dtype == ACL_COMPLEX64) {
                    EXPECT_FLOAT_EQ(parts[1], expected[index].imag());
                }
            } else if (dtype == ACL_INT32) {
                int32_t value = 0;
                ASSERT_TRUE(spmm_test::SecureCopy(&value, sizeof(value), element, sizeof(value)));
                EXPECT_EQ(value, static_cast<int32_t>(expected[index].real()));
            } else {
                uint16_t value = 0;
                ASSERT_TRUE(spmm_test::SecureCopy(&value, sizeof(value), element, sizeof(value)));
                EXPECT_FLOAT_EQ(dtype == ACL_FLOAT16 ? spmm_test::HalfToFloat(value) :
                    spmm_test::Bf16ToFloat(value), expected[index].real());
            }
        }
    }

    void Execute(const RunOptions &options);
    bool PrepareDenseInputs();
    bool PrepareOutput();
    bool PrepareWorkspace(aclsparseSpMatDescr_t matA, aclsparseOperation_t opA);
    void CheckScalarUpdates(aclsparseSpMatDescr_t matA,
        const std::vector<int32_t> &columns, const std::vector<Complex> &values,
        aclsparseOperation_t opA, const std::vector<uint8_t> &hostScalar,
        void *deviceScalar);
    void CheckAlg3Rejection(aclsparseSpMatDescr_t matA, aclsparseOperation_t opA);
    void CheckAlternation(aclsparseSpMatDescr_t first, aclsparseSpMatDescr_t second,
        void *deviceValues, const std::vector<int32_t> &columns,
        const std::vector<int32_t> &other, const std::vector<Complex> &values,
        aclsparseOperation_t opA);
    void RunAndCheck(aclsparseSpMatDescr_t matA, const std::vector<int32_t> &columns,
                     const std::vector<Complex> &values, aclsparseOperation_t opA,
                     bool preprocess);
    std::vector<Complex> Golden(const std::vector<int32_t> &columns,
                                const std::vector<Complex> &values,
                                aclsparseOperation_t opA) const;

    aclsparseHandle_t handle_ = nullptr;
    std::vector<void *> buffers_;
    std::vector<aclsparseSpMatDescr_t> sparse_;
    std::vector<aclsparseDnMatDescr_t> dense_;
    RunOptions options_;
    int64_t columnCount_ = kColumns;
    aclDataType outputType_ = ACL_FLOAT;
    aclDataType computeType_ = ACL_FLOAT;
    aclsparseDnMatDescr_t matB_ = nullptr;
    aclsparseDnMatDescr_t matC_ = nullptr;
    aclsparseDnMatDescr_t matOut_ = nullptr;
    void *dC_ = nullptr;
    void *dOut_ = nullptr;
    void *workspace_ = nullptr;
    const void *alpha_ = nullptr;
    const void *beta_ = nullptr;
    int64_t cLd_ = 0;
    std::vector<Complex> denseB_;
    std::vector<Complex> initialC_;
};

std::vector<Complex> SpmmTransposeTest::Golden(const std::vector<int32_t> &columns,
    const std::vector<Complex> &values, aclsparseOperation_t opA) const
{
    auto result = initialC_;
    if (options_.empty) return result;
    for (int32_t row = 0; row < kRows; ++row) {
        for (int32_t position = row * 2; position < row * 2 + 2; ++position) {
            const Complex value = opA == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ?
                std::conj(values[position]) : values[position];
            for (int32_t column = 0; column < kN; ++column) {
                const int32_t outputRow = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? row : columns[position];
                const int32_t inputRow = opA == ACL_SPARSE_OP_NON_TRANSPOSE ? columns[position] : row;
                result[DenseIndex(outputRow, column, cLd_, options_.cOrder)] +=
                    value * denseB_[inputRow * kN + column];
            }
        }
    }
    return result;
}

void SpmmTransposeTest::RunAndCheck(aclsparseSpMatDescr_t matA,
    const std::vector<int32_t> &columns, const std::vector<Complex> &values,
    aclsparseOperation_t opA, bool preprocess)
{
    ASSERT_TRUE(Upload(dC_, initialC_, outputType_));
    if (preprocess) {
        ASSERT_EQ(aclsparseSpMMPreprocess(handle_, opA, options_.opB, alpha_, matA,
            matB_, beta_, matOut_, computeType_, options_.algorithm, workspace_),
            ACL_SPARSE_STATUS_SUCCESS);
    }
    const auto status = options_.separateOutput ? aclsparseSpMMPrivateOutOfPlace(
        handle_, opA, options_.opB, alpha_, matA, matB_, beta_, matC_, matOut_,
        computeType_, options_.algorithm, workspace_) :
        aclsparseSpMM(handle_, opA, options_.opB, alpha_, matA, matB_, beta_, matC_,
            computeType_, options_.algorithm, workspace_);
    ASSERT_EQ(status, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    CheckResult(options_.separateOutput ? dOut_ : dC_, Golden(columns, values, opA), outputType_);
    if (options_.separateOutput) CheckResult(dC_, initialC_, outputType_);
}

bool SpmmTransposeTest::PrepareDenseInputs()
{
    const aclDataType dtype = std::get<0>(GetParam());
    const int64_t bRows = options_.opB == ACL_SPARSE_OP_NON_TRANSPOSE ? kRows : kN;
    const int64_t bCols = options_.opB == ACL_SPARSE_OP_NON_TRANSPOSE ? kN : kRows;
    const int64_t bLd = (options_.bOrder == ACL_SPARSE_ORDER_ROW ? bCols : bRows) + 1;
    void *dB = nullptr;
    if (!CreateDense(bRows, bCols, bLd, options_.bOrder, dtype, &matB_, &dB)) return false;
    std::vector<Complex> storageB((options_.bOrder == ACL_SPARSE_ORDER_ROW ? bRows : bCols) * bLd);
    denseB_.resize(kRows * kN);
    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t col = 0; col < kN; ++col) {
            const Complex value(static_cast<float>((row + col) % 3) - 1,
                dtype == ACL_COMPLEX64 ? static_cast<float>((row + 2 * col) % 3) - 1 : 0);
            denseB_[row * kN + col] = value;
            const Complex stored = options_.opB == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ? std::conj(value) : value;
            storageB[options_.opB == ACL_SPARSE_OP_NON_TRANSPOSE ? DenseIndex(row, col, bLd, options_.bOrder) :
                DenseIndex(col, row, bLd, options_.bOrder)] = stored;
        }
    }
    return Upload(dB, storageB, dtype);
}

bool SpmmTransposeTest::PrepareOutput()
{
    cLd_ = (options_.cOrder == ACL_SPARSE_ORDER_ROW ? kN : columnCount_) + 1;
    if (!CreateDense(columnCount_, kN, cLd_, options_.cOrder, outputType_, &matC_, &dC_)) return false;
    matOut_ = matC_;
    if (options_.separateOutput && !CreateDense(columnCount_, kN, cLd_, options_.cOrder,
        outputType_, &matOut_, &dOut_)) return false;
    initialC_.assign((options_.cOrder == ACL_SPARSE_ORDER_ROW ? columnCount_ : kN) * cLd_, Complex(1, 0));
    return !options_.separateOutput || Upload(dOut_, initialC_, outputType_);
}

bool SpmmTransposeTest::PrepareWorkspace(aclsparseSpMatDescr_t matA,
    aclsparseOperation_t opA)
{
    size_t size = 0;
    if (aclsparseSpMMGetBufferSize(handle_, opA, options_.opB, alpha_, matA, matB_, beta_,
        matOut_, computeType_, options_.algorithm, &size) != ACL_SPARSE_STATUS_SUCCESS) return false;
    if (options_.square) {
        size_t normalSize = 0;
        if (aclsparseSpMMGetBufferSize(handle_, ACL_SPARSE_OP_NON_TRANSPOSE, options_.opB,
            alpha_, matA, matB_, beta_, matOut_, computeType_, options_.algorithm,
            &normalSize) != ACL_SPARSE_STATUS_SUCCESS) return false;
        size = std::max(size, normalSize);
    }
    workspace_ = Allocate(size);
    return workspace_ != nullptr;
}

void SpmmTransposeTest::Execute(const RunOptions &options)
{
    options_ = options;
    columnCount_ = options.square ? kRows : kColumns;
    const aclDataType dtype = std::get<0>(GetParam());
    outputType_ = dtype == ACL_INT8 ? ACL_INT32 : (options.mixedOutput ? ACL_FLOAT : dtype);
    computeType_ = dtype == ACL_COMPLEX64 ? ACL_COMPLEX64 : (dtype == ACL_INT8 ? ACL_INT32 : ACL_FLOAT);
    ASSERT_EQ(aclsparseCreate(&handle_), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseSetStream(handle_, stream_), ACL_SPARSE_STATUS_SUCCESS);
    const auto values = Values(0);
    void *dValues = Allocate(kNnz * ValueSize(dtype));
    ASSERT_NE(dValues, nullptr);
    ASSERT_TRUE(Upload(dValues, values, dtype));
    aclsparseSpMatDescr_t first = nullptr, second = nullptr;
    const std::vector<int32_t> columns = options.square ?
        std::vector<int32_t>{0, 2, 1, 2, 0, 1} : std::vector<int32_t>{0, 3, 1, 3, 0, 2};
    const std::vector<int32_t> other = options.square ?
        std::vector<int32_t>{1, 2, 0, 1, 0, 2} : std::vector<int32_t>{1, 4, 0, 2, 1, 4};
    ASSERT_TRUE(CreateSparse(columns, dValues, &first));
    ASSERT_TRUE(CreateSparse(other, dValues, &second));
    ASSERT_TRUE(PrepareDenseInputs());
    ASSERT_TRUE(PrepareOutput());
    const auto scalar = Encode({Complex(1, 0)}, computeType_);
    void *dScalar = Allocate(scalar.size());
    ASSERT_NE(dScalar, nullptr);
    ASSERT_EQ(aclrtMemcpy(dScalar, scalar.size(), scalar.data(), scalar.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    alpha_ = options.deviceScalars ? dScalar : scalar.data();
    beta_ = alpha_;
    ASSERT_EQ(aclsparseSetPointerMode(handle_, options.deviceScalars ? ACL_SPARSE_POINTER_MODE_DEVICE :
        ACL_SPARSE_POINTER_MODE_HOST), ACL_SPARSE_STATUS_SUCCESS);
    const auto opA = std::get<1>(GetParam());
    ASSERT_TRUE(PrepareWorkspace(first, opA));
    CheckAlternation(first, second, dValues, columns, other, values, opA);
    CheckScalarUpdates(first, columns, values, opA, scalar, dScalar);
    CheckAlg3Rejection(first, opA);
}

void SpmmTransposeTest::CheckAlternation(aclsparseSpMatDescr_t first,
    aclsparseSpMatDescr_t second, void *deviceValues, const std::vector<int32_t> &columns,
    const std::vector<int32_t> &other, const std::vector<Complex> &values,
    aclsparseOperation_t opA)
{
    if (options_.square) {
        RunAndCheck(first, columns, values, ACL_SPARSE_OP_NON_TRANSPOSE, options_.preprocess);
    }
    RunAndCheck(first, columns, values, opA, options_.preprocess);
    RunAndCheck(second, other, values, opA, options_.preprocess);
    RunAndCheck(first, columns, values, opA, false);
    const auto updated = Values(2);
    ASSERT_TRUE(Upload(deviceValues, updated, std::get<0>(GetParam())));
    RunAndCheck(first, columns, updated, opA, false);
    const std::vector<Complex> zero(kNnz);
    ASSERT_TRUE(Upload(deviceValues, zero, std::get<0>(GetParam())));
    RunAndCheck(first, columns, zero, opA, false);
    ASSERT_TRUE(Upload(deviceValues, values, std::get<0>(GetParam())));
    RunAndCheck(first, columns, values, opA == ACL_SPARSE_OP_TRANSPOSE ?
        ACL_SPARSE_OP_CONJUGATE_TRANSPOSE : ACL_SPARSE_OP_TRANSPOSE, false);
    RunAndCheck(first, columns, values, opA, false);
    if (options_.square) {
        RunAndCheck(first, columns, values, ACL_SPARSE_OP_NON_TRANSPOSE, false);
    }
}

void SpmmTransposeTest::CheckAlg3Rejection(aclsparseSpMatDescr_t matA,
    aclsparseOperation_t opA)
{
    size_t bytes = 0;
    EXPECT_EQ(aclsparseSpMMGetBufferSize(handle_, opA, options_.opB, alpha_, matA, matB_, beta_,
        matOut_, computeType_, ACL_SPARSE_SPMM_CSR_ALG3, &bytes), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(aclsparseSpMMPreprocess(handle_, opA, options_.opB, alpha_, matA, matB_, beta_,
        matOut_, computeType_, ACL_SPARSE_SPMM_CSR_ALG3, workspace_), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(aclsparseSpMM(handle_, opA, options_.opB, alpha_, matA, matB_, beta_,
        matOut_, computeType_, ACL_SPARSE_SPMM_CSR_ALG3, workspace_), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

void SpmmTransposeTest::CheckScalarUpdates(aclsparseSpMatDescr_t matA,
    const std::vector<int32_t> &columns, const std::vector<Complex> &values,
    aclsparseOperation_t opA, const std::vector<uint8_t> &hostScalar,
    void *deviceScalar)
{
    const auto doubled = Encode({Complex(2, 0)}, computeType_);
    ASSERT_EQ(aclrtMemcpy(deviceScalar, doubled.size(), doubled.data(), doubled.size(),
        ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    alpha_ = options_.deviceScalars ? deviceScalar : doubled.data();
    beta_ = alpha_;
    ASSERT_TRUE(Upload(dC_, initialC_, outputType_));
    ASSERT_EQ(aclsparseSpMM(handle_, opA, options_.opB, alpha_, matA, matB_, beta_,
        matC_, computeType_, options_.algorithm, workspace_), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    auto expected = Golden(columns, values, opA);
    for (int64_t row = 0; row < columnCount_; ++row) {
        for (int64_t col = 0; col < kN; ++col) {
            expected[DenseIndex(row, col, cLd_, options_.cOrder)] *= 2.0F;
        }
    }
    CheckResult(dC_, expected, outputType_);
    ASSERT_EQ(aclrtMemcpy(deviceScalar, hostScalar.size(), hostScalar.data(), hostScalar.size(),
        ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    alpha_ = options_.deviceScalars ? deviceScalar : hostScalar.data();
    beta_ = alpha_;
    ASSERT_EQ(aclsparseSetPointerMode(handle_, options_.deviceScalars ?
        ACL_SPARSE_POINTER_MODE_HOST : ACL_SPARSE_POINTER_MODE_DEVICE), ACL_SPARSE_STATUS_SUCCESS);
    alpha_ = options_.deviceScalars ? hostScalar.data() : deviceScalar;
    beta_ = alpha_;
    RunAndCheck(matA, columns, values, opA, false);
}

TEST_P(SpmmTransposeTest, RectangularOwnershipAndLiveValues)
{
    RunOptions options;
    options.preprocess = std::get<2>(GetParam()) == 0;
    options.deviceScalars = std::get<1>(GetParam()) == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    Execute(options);
}

TEST_P(SpmmTransposeTest, DenseTransposeAndColumnMajor)
{
    RunOptions options;
    options.opB = std::get<2>(GetParam()) == 0 ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    options.bOrder = ACL_SPARSE_ORDER_COL;
    options.cOrder = ACL_SPARSE_ORDER_COL;
    options.deviceScalars = true;
    options.algorithm = ACL_SPARSE_SPMM_CSR_ALG1;
    Execute(options);
}

TEST_P(SpmmTransposeTest, SeparateInitialCAndMixedOutput)
{
    const auto dtype = std::get<0>(GetParam());
    RunOptions options;
    options.separateOutput = dtype == ACL_FLOAT || dtype == ACL_COMPLEX64;
    options.mixedOutput = dtype == ACL_FLOAT16 || dtype == ACL_BF16;
    options.preprocess = true;
    options.opB = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    options.algorithm = ACL_SPARSE_SPMM_ALG_DEFAULT;
    Execute(options);
}

TEST_P(SpmmTransposeTest, NonTransposeAndTransposeWorkspaceSwitch)
{
    RunOptions options;
    options.square = true;
    options.algorithm = std::get<0>(GetParam()) == ACL_FLOAT ?
        ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG : ACL_SPARSE_SPMM_CSR_ALG2;
    options.preprocess = std::get<2>(GetParam()) == 0;
    Execute(options);
}

TEST_P(SpmmTransposeTest, EmptySparseProduct)
{
    RunOptions options;
    options.empty = true;
    options.preprocess = std::get<2>(GetParam()) == 0;
    Execute(options);
}

INSTANTIATE_TEST_SUITE_P(Arch35, SpmmTransposeTest, ::testing::Combine(
    ::testing::Values(ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_COMPLEX64, ACL_INT8),
    ::testing::Values(ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE),
    ::testing::Values(0, 1)));

} // namespace
