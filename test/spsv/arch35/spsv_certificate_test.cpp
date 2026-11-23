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

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>
#include "test_common.h"
#include "spsv_npu_wrapper.h"
#include "../../../sparse/spsv/arch35/op_kernel/recovery/spsv_dyadic_layout.h"

namespace sparse_test
{
namespace
{
using Complex = std::complex<float>;
using WideComplex = std::complex<double>;

struct CertificateCase {
    CsrMatrix matrix;
    std::vector<Complex> values;
    std::vector<Complex> input;
    std::vector<WideComplex> expected;
};

void ExpectReplayComponent(float want, float got, float absolute = 1.e-6f, float relative = 1.e-5f)
{
    if (std::isnan(want)) {
        EXPECT_TRUE(std::isnan(got));
    } else if (std::isinf(want)) {
        EXPECT_EQ(got, want);
    } else {
        EXPECT_NEAR(got, want, std::max(absolute, std::abs(want) * relative));
    }
}

template <typename T>
void ExpectCertificateSolution(const std::vector<Complex>& actual,
                               const std::vector<std::complex<T>>& expected,
                               float absolute = 0.f, float relative = 0.f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t row = 0; row < actual.size(); ++row) {
        SCOPED_TRACE(row);
        ExpectReplayComponent(static_cast<float>(expected[row].real()), actual[row].real(), absolute, relative);
        ExpectReplayComponent(static_cast<float>(expected[row].imag()), actual[row].imag(), absolute, relative);
    }
}

CertificateCase MakeUnitCertificateCase(int32_t m, bool lower)
{
    CertificateCase out;
    out.matrix.rows = out.matrix.cols = m;
    out.matrix.rowOffsets = {0};
    out.input.resize(m);
    out.expected.resize(m);
    // The signed recurrence is contractive (roots have magnitude sqrt(0.75)),
    // although an absolute-coefficient error recurrence grows exponentially.
    for (int32_t r = 0; r < m; ++r) {
        for (int32_t slot = 0; slot < 3; ++slot) {
            int32_t distance = lower ? 2 - slot : slot;
            int32_t c = lower ? r - distance : r + distance;
            if (c < 0 || c >= m) {
                continue;
            }
            float a = distance == 0 ? 1.0f : distance == 1 ? -1.5f : 0.75f;
            out.matrix.colIndices.push_back(c);
            out.matrix.values.push_back(a);
            out.values.emplace_back(a, 0.0f);
        }
        out.matrix.rowOffsets.push_back(static_cast<int32_t>(out.values.size()));
        out.input[r] = Complex((r % 7 - 3) * 0.125f, (r % 11 - 5) * 0.0625f);
    }
    out.matrix.nnz = static_cast<int64_t>(out.values.size());
    for (int32_t step = 0; step < m; ++step) {
        int32_t r = lower ? step : m - 1 - step;
        WideComplex y(out.input[r].real(), out.input[r].imag());
        for (int32_t p = out.matrix.rowOffsets[r]; p < out.matrix.rowOffsets[r + 1]; ++p) {
            int32_t c = out.matrix.colIndices[p];
            if (c != r) {
                y -= WideComplex(out.values[p].real(), out.values[p].imag()) * out.expected[c];
            }
        }
        out.expected[r] = y;
    }
    return out;
}
}  // namespace

class SpsvCertificateTest : public ::testing::Test {
protected:
    inline static std::unique_ptr<AclEnvScope> sharedEnv;
    inline static std::unique_ptr<HandleManager> sharedHandle;
    static void SetUpTestSuite()
    {
        sharedEnv = std::make_unique<AclEnvScope>();
        sharedHandle = std::make_unique<HandleManager>();
        sharedHandle->setStream(sharedEnv->stream());
    }
    static void TearDownTestSuite()
    {
        sharedHandle.reset();
        sharedEnv.reset();
    }
    AclEnvScope& env{*sharedEnv};
    HandleManager& handle{*sharedHandle};

    template <typename T>
    void CheckCertificate(const CertificateCase& test, const std::vector<std::complex<T>>& expected,
                          const SpSVNpuConfig& cfg, float absolute = 0.f, float relative = 0.f)
    {
        auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), test.matrix, test.values,
                                            test.input, Complex(1.f, 0.f), cfg);
        ExpectCertificateSolution(result.y, expected, absolute, relative);
    }
};

TEST_F(SpsvCertificateTest, ReviewComplexUnitBandedCertificate)
{
    for (int32_t m : {127, 128, 129, 257, 513}) {
        for (bool lower : {false, true}) {
            auto test = MakeUnitCertificateCase(m, lower);
            for (bool inPlace : {false, true}) {
                SpSVNpuConfig cfg;
                cfg.lower = lower;
                cfg.unitDiag = true;
                cfg.inPlace = inPlace;
                SCOPED_TRACE(::testing::Message() << m << ':' << lower << ':' << inPlace);
                CheckCertificate(test, test.expected, cfg, 0x1p-18f);
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexNonUnitBandedCertificate)
{
    const Complex diagonal(3.0f, 0.5f);
    for (int32_t m : {128, 129, 159, 160, 161, 257, 513}) {
        for (bool lower : {false, true}) {
            auto test = MakeUnitCertificateCase(m, lower);
            for (size_t p = 0; p < test.values.size(); ++p) {
                test.values[p] *= diagonal;
                test.matrix.values[p] = test.values[p].real();
            }
            for (auto operation :
                 {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE}) {
                bool transposed = operation != ACL_SPARSE_OP_NON_TRANSPOSE;
                auto reference = MakeUnitCertificateCase(m, transposed ? !lower : lower);
                WideComplex divisor(diagonal.real(), diagonal.imag());
                if (operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
                    divisor = std::conj(divisor);
                }
                for (bool inPlace : {false, true}) {
                    SpSVNpuConfig cfg;
                    cfg.lower = lower;
                    cfg.opA = operation;
                    cfg.inPlace = inPlace;
                    auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), test.matrix, test.values,
                                                        test.input, Complex(1.0f, 0.0f), cfg);
                    for (int32_t r = 0; r < m; ++r) {
                        SCOPED_TRACE(::testing::Message()
                                     << m << ":" << lower << ":" << operation << ":" << inPlace << ":" << r);
                        WideComplex expected = reference.expected[r] / divisor;
                        EXPECT_NEAR(result.y[r].real(), expected.real(), 0x1p-18);
                        EXPECT_NEAR(result.y[r].imag(), expected.imag(), 0x1p-18);
                    }
                }
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexNonUnitPrefixMidpointCorrection)
{
    // This recurrence crosses an FP32 midpoint after the shadow solve.
    // Independent 100/160-digit solves agree on the expected rounded result.
    constexpr int32_t rows = 5376;
    constexpr int32_t seed = 20261817;
    CsrMatrix matrix;
    matrix.rows = matrix.cols = rows;
    matrix.rowOffsets = {0};
    std::vector<Complex> values;
    std::vector<Complex> input(rows);
    auto value = [](int32_t i) { return Complex(0.25f + (i % 17) * 0.01f, 0.1f + (i % 7) * 0.01f); };
    int32_t at = 0;
    for (int32_t r = 0; r < rows; ++r) {
        int32_t distance = std::min(rows - 1 - r, r < 45 ? 10 : 9);
        for (int32_t c = r; c <= r + distance; ++c) {
            Complex a = value(at + 1 + seed);
            matrix.colIndices.push_back(c);
            matrix.values.push_back(a.real());
            values.push_back(a);
            ++at;
        }
        matrix.rowOffsets.push_back(at);
        input[r] = value(r + 101 + seed);
    }
    matrix.nnz = at;
    ASSERT_EQ(at, 53760);
    for (bool inPlace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.lower = false;
        cfg.opA = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
        cfg.inPlace = inPlace;
        auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), matrix, values, input, value(1), cfg);
        EXPECT_EQ(result.y[3296].imag(), -19554.275390625f);
    }
}
namespace
{
float BorderCertificateCoefficient(int32_t step, int32_t source, bool head, int32_t tailRows,
                                   int32_t independentRows)
{
    constexpr int32_t rows = 1101;
    constexpr int32_t headRows = 35;
    int32_t distance = step - source;
    float coefficient = distance == 0 ? 1.0f : distance == 1 ? -1.5f : distance == 2 ? 0.75f : 0.0f;
    if (distance > 2 && head && ((step < headRows) || source < 2)) {
        coefficient = 0x1p-10f;
    }
    if (distance > 2 && !head && step >= rows - tailRows && source % 17 == 0) {
        coefficient = 0x1p-12f;
    }
    if (step < independentRows && distance > 0) {
        coefficient = 0.0f;
    }
    return coefficient;
}

CertificateCase MakeBorderCertificateCase(bool head, bool lower, bool unit, int32_t tailRows = 45,
                                          int32_t independentRows = 1)
{
    constexpr int32_t rows = 1101;
    CertificateCase out;
    out.matrix.rows = out.matrix.cols = rows;
    out.matrix.rowOffsets = {0};
    out.input.resize(rows);
    Complex diagonal = unit ? Complex(1.0f, 0.0f) : Complex(3.0f, 0.5f);
    for (int32_t row = 0; row < rows; ++row) {
        int32_t step = lower ? row : rows - 1 - row;
        std::vector<std::pair<int32_t, Complex>> entries;
        for (int32_t source = 0; source <= step; ++source) {
            float coefficient = BorderCertificateCoefficient(step, source, head, tailRows, independentRows);
            if (coefficient == 0.0f) {
                continue;
            }
            int32_t column = lower ? source : rows - 1 - source;
            entries.emplace_back(column, coefficient * diagonal);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& entry : entries) {
            out.matrix.colIndices.push_back(entry.first);
            out.matrix.values.push_back(entry.second.real());
            out.values.push_back(entry.second);
        }
        out.matrix.rowOffsets.push_back(static_cast<int32_t>(out.values.size()));
        out.input[row] = Complex((row % 7 - 3) * 0.125f, (row % 11 - 5) * 0.0625f);
    }
    out.matrix.nnz = static_cast<int64_t>(out.values.size());
    return out;
}

std::vector<WideComplex> SolveBorderReference(const CertificateCase& test, bool lower, bool unit,
                                              bool adjoint)
{
    int32_t rows = static_cast<int32_t>(test.input.size());
    std::vector<std::vector<std::pair<int32_t, WideComplex>>> equations(rows);
    for (int32_t row = 0; row < rows; ++row) {
        for (int32_t p = test.matrix.rowOffsets[row]; p < test.matrix.rowOffsets[row + 1]; ++p) {
            int32_t column = test.matrix.colIndices[p];
            WideComplex value(test.values[p].real(), test.values[p].imag());
            if (adjoint) {
                equations[column].emplace_back(row, std::conj(value));
            } else {
                equations[row].emplace_back(column, value);
            }
        }
    }
    std::vector<WideComplex> result(rows);
    bool forward = lower != adjoint;
    for (int32_t step = 0; step < rows; ++step) {
        int32_t row = forward ? step : rows - 1 - step;
        WideComplex numerator =
            WideComplex(1.0, 0.125) * WideComplex(test.input[row].real(), test.input[row].imag());
        WideComplex diagonal(1.0, 0.0);
        for (const auto& entry : equations[row]) {
            if (entry.first == row) {
                if (!unit) {
                    diagonal = entry.second;
                }
            } else {
                numerator -= entry.second * result[entry.first];
            }
        }
        result[row] = numerator / diagonal;
    }
    return result;
}

CertificateCase AdjointBorderCase(const CertificateCase& test)
{
    CertificateCase out;
    out.matrix.rows = test.matrix.rows;
    out.matrix.cols = test.matrix.cols;
    out.matrix.nnz = test.matrix.nnz;
    out.matrix.rowOffsets = {0};
    out.input = test.input;
    std::vector<std::vector<std::pair<int32_t, Complex>>> entries(test.input.size());
    for (int32_t row = 0; row < test.matrix.rows; ++row) {
        for (int32_t p = test.matrix.rowOffsets[row]; p < test.matrix.rowOffsets[row + 1]; ++p) {
            entries[test.matrix.colIndices[p]].emplace_back(row, std::conj(test.values[p]));
        }
    }
    for (const auto& row : entries) {
        for (const auto& entry : row) {
            out.matrix.colIndices.push_back(entry.first);
            out.matrix.values.push_back(entry.second.real());
            out.values.push_back(entry.second);
        }
        out.matrix.rowOffsets.push_back(static_cast<int32_t>(out.values.size()));
    }
    return out;
}

void CheckBorderCertificate(aclsparseHandle_t handle, aclrtStream stream, bool head)
{
    for (bool lower : {false, true}) {
        {
            // Exercise the dispatch branches that own these structures:
            // UNIT head/anchor and NON_UNIT banded-prefix/arbitrary-tail.
            bool unit = head;
            auto original = MakeBorderCertificateCase(head, lower, unit);
            for (bool adjoint : {false, true}) {
                auto test = adjoint ? AdjointBorderCase(original) : original;
                bool storedLower = lower != adjoint;
                auto expected = SolveBorderReference(test, storedLower, unit, adjoint);
                for (bool inPlace : {false, true}) {
                    SpSVNpuConfig cfg;
                    cfg.lower = storedLower;
                    cfg.unitDiag = unit;
                    cfg.inPlace = inPlace;
                    cfg.opA = adjoint ? ACL_SPARSE_OP_CONJUGATE_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
                    auto result = SpSVNpuWrapperComplex(handle, stream, test.matrix, test.values,
                                                        test.input, Complex(1.0f, 0.125f), cfg);
                    for (size_t row = 0; row < expected.size(); ++row) {
                        SCOPED_TRACE(::testing::Message() << head << ":" << lower << ":" << unit << ":"
                                                          << adjoint << ":" << inPlace << ":" << row);
                        EXPECT_NEAR(result.y[row].real(), expected[row].real(), 0x1p-18);
                        EXPECT_NEAR(result.y[row].imag(), expected[row].imag(), 0x1p-18);
                    }
                }
            }
        }
    }
}
}  // namespace

TEST_F(SpsvCertificateTest, ReviewComplexBorderedHeadPartialBlocks)
{
    CheckBorderCertificate(handle.get(), env.stream(), true);
}

TEST_F(SpsvCertificateTest, ReviewComplexBorderedTailPartialBlocks)
{
    CheckBorderCertificate(handle.get(), env.stream(), false);
}

TEST_F(SpsvCertificateTest, ReviewComplexPreconditionedTailPartialBlocks)
{
    for (int32_t tailRows : {129, 191, 303}) {
        for (bool lower : {false, true}) {
            auto original = MakeBorderCertificateCase(false, lower, false, tailRows, 128);
            for (bool adjoint : {false, true}) {
                auto test = adjoint ? AdjointBorderCase(original) : original;
                bool storedLower = lower != adjoint;
                auto expected = SolveBorderReference(test, storedLower, false, adjoint);
                for (bool inPlace : {false, true}) {
                    SpSVNpuConfig cfg;
                    cfg.lower = storedLower;
                    cfg.unitDiag = false;
                    cfg.inPlace = inPlace;
                    cfg.opA = adjoint ? ACL_SPARSE_OP_CONJUGATE_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
                    auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), test.matrix, test.values,
                                                        test.input, Complex(1.0f, 0.125f), cfg);
                    for (size_t row = 0; row < expected.size(); ++row) {
                        SCOPED_TRACE(::testing::Message() << tailRows << ':' << lower << ':' << adjoint << ':'
                                                          << inPlace << ':' << row);
                        ASSERT_NEAR(result.y[row].real(), expected[row].real(), 0x1p-18);
                        ASSERT_NEAR(result.y[row].imag(), expected[row].imag(), 0x1p-18);
                    }
                }
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewDyadicDigitOffsetInt64Boundary)
{
    constexpr int64_t beforeOverflowRow = (1LL << 26) - 1;
    constexpr int64_t overflowRow = 1LL << 26;
    EXPECT_EQ(SpsvDyadicDigitOffset(beforeOverflowRow, 0), 2147483616LL);
    EXPECT_EQ(SpsvDyadicDigitOffset(beforeOverflowRow, kSpsvDyadicLimbs - 1), 2147483647LL);
    EXPECT_EQ(SpsvDyadicDigitOffset(overflowRow, 0), 2147483648LL);
    EXPECT_EQ(SpsvDyadicDigitOffset(overflowRow, kSpsvDyadicLimbs - 1), 2147483679LL);
}
TEST_F(SpsvCertificateTest, ReviewComplexDeepCandidateCertificate)
{
    for (int32_t m : {1023, 1024, 1025, 4097}) {
        for (bool lower : {false, true}) {
            for (bool nonUnit : {false, true}) {
                auto test = MakeUnitCertificateCase(m, lower);
                if (nonUnit) {
                    for (auto& value : test.values) {
                        value *= 2.0f;
                    }
                    for (auto& value : test.matrix.values) {
                        value *= 2.0f;
                    }
                    for (auto& value : test.input) {
                        value *= 2.0f;
                    }
                }
                for (bool inPlace : {false, true}) {
                    SpSVNpuConfig cfg;
                    cfg.lower = lower;
                    cfg.unitDiag = !nonUnit;
                    cfg.inPlace = inPlace;
                    auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), test.matrix, test.values,
                                                        test.input, Complex(1.0f, 0.0f), cfg);
                    for (int32_t r = 0; r < m; ++r) {
                        SCOPED_TRACE(::testing::Message()
                                     << m << ':' << lower << ':' << nonUnit << ':' << inPlace << ':' << r);
                        ASSERT_NEAR(result.y[r].real(), test.expected[r].real(), 0x1p-18);
                        ASSERT_NEAR(result.y[r].imag(), test.expected[r].imag(), 0x1p-18);
                    }
                }
            }
        }
    }
}

CertificateCase MakeOverflowCancellationCase(bool lower)
{
    constexpr int32_t m = 1024;
    auto test = MakeUnitCertificateCase(m, lower);
    test.values.clear();
    test.matrix.values.clear();
    test.matrix.colIndices.clear();
    test.matrix.rowOffsets = {0};
    for (int32_t r = 0; r < m; ++r) {
        int32_t step = lower ? r : m - 1 - r;
        int32_t previous = lower ? r - 1 : r + 1;
        Complex coefficient = step == m - 1 ? Complex(1.0f, 1.0f) : Complex(-0.5f, 0.0f);
        Complex diagonal(step == m - 1 ? 2.0f : 1.0f, 0.0f);
        auto append = [&](int32_t column, Complex value) {
            test.matrix.colIndices.push_back(column);
            test.matrix.values.push_back(value.real());
            test.values.push_back(value);
        };
        if (lower && step > 0) {
            append(previous, coefficient);
        }
        append(r, diagonal);
        if (!lower && step > 0) {
            append(previous, coefficient);
        }
        test.matrix.rowOffsets.push_back(static_cast<int32_t>(test.values.size()));
        test.input[r] = step == m - 2   ? Complex(3.0e38f, 3.0e38f)
                        : step == m - 1 ? Complex(0.0f, 3.0e38f)
                                        : Complex(0.5f, 0.0f);
    }
    test.matrix.nnz = static_cast<int64_t>(test.values.size());
    return test;
}

std::vector<std::complex<long double>> OverflowCancellationReference(const CertificateCase& test, bool lower)
{
    constexpr int32_t m = 1024;
    using Exact = std::complex<long double>;
    std::vector<Exact> expected(m);
    for (int32_t step = 0; step < m; ++step) {
        int32_t r = lower ? step : m - 1 - step;
        Exact value(test.input[r].real(), test.input[r].imag());
        Exact diagonal(1.0L, 0.0L);
        for (int32_t p = test.matrix.rowOffsets[r]; p < test.matrix.rowOffsets[r + 1]; ++p) {
            int32_t c = test.matrix.colIndices[p];
            Exact a(test.values[p].real(), test.values[p].imag());
            if (c == r) {
                diagonal = a;
            } else {
                value -= a * expected[c];
            }
        }
        expected[r] = value / diagonal;
    }
    // The final real component cancels two values near 3e38. A native
    // long double oracle loses the earlier half-unit and gives zero.
    // For the exact dyadic recurrence the prefix is 1-2^(-step-1),
    // so this component is -1/4 + 2^-1024 and rounds to -1/4.
    // The final imaginary component rounds to -B/2.
    int32_t finalRow = lower ? m - 1 : 0;
    expected[finalRow] = Exact(-0.25L, -0.5L * static_cast<long double>(3.0e38f));
    return expected;
}

TEST_F(SpsvCertificateTest, ReviewComplexDeepIntermediateOverflowCancellation)
{
    for (bool lower : {false, true}) {
        auto test = MakeOverflowCancellationCase(lower);
        auto expected = OverflowCancellationReference(test, lower);
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.lower = lower;
            cfg.unitDiag = false;
            cfg.inPlace = inPlace;
            SCOPED_TRACE(::testing::Message() << lower << ':' << inPlace);
            CheckCertificate(test, expected, cfg, 0x1p-18f, 4.f * std::numeric_limits<float>::epsilon());
        }
    }
}

CertificateCase MakeSingletonWideTransitionCase(bool lower)
{
    constexpr int32_t m = 1400;
    CertificateCase test;
    test.matrix.rows = test.matrix.cols = m;
    test.matrix.rowOffsets = {0};
    test.input.resize(m);
    test.expected.resize(m);
    std::vector<int32_t> predecessor(m, -1);
    for (int32_t begin = 0; begin < m;) {
        int32_t width = begin % 117 == 0 && begin > 0 ? 17 : 1;
        width = std::min(width, m - begin);
        for (int32_t j = 0; j < width; ++j) {
            predecessor[begin + j] = begin - 1;
        }
        begin += width;
    }
    for (int32_t r = 0; r < m; ++r) {
        int32_t at = lower ? r : m - 1 - r;
        int32_t previous = predecessor[at];
        int32_t c = previous < 0 ? -1 : lower ? previous : m - 1 - previous;
        auto append = [&](int32_t column, Complex value) {
            test.matrix.colIndices.push_back(column);
            test.matrix.values.push_back(value.real());
            test.values.push_back(value);
        };
        if (lower && c >= 0) {
            append(c, Complex(-0.5f, 0.125f));
        }
        append(r, Complex(1.0f, 0.0f));
        if (!lower && c >= 0) {
            append(c, Complex(-0.5f, 0.125f));
        }
        test.matrix.rowOffsets.push_back(static_cast<int32_t>(test.values.size()));
        test.input[r] = Complex((at % 13 - 6) * 0.125f, (at % 7 - 3) * 0.25f);
    }
    test.matrix.nnz = static_cast<int64_t>(test.values.size());
    for (int32_t step = 0; step < m; ++step) {
        int32_t r = lower ? step : m - 1 - step;
        test.expected[r] = WideComplex(test.input[r].real(), test.input[r].imag());
        if (predecessor[step] >= 0) {
            int32_t c = lower ? predecessor[step] : m - 1 - predecessor[step];
            test.expected[r] -= WideComplex(-0.5, 0.125) * test.expected[c];
        }
    }
    return test;
}

CertificateCase MakeDistributedComplexRows(bool lower, bool unit, float magnitude)
{
    constexpr int32_t independent = 2048, m = independent + 2;
    CertificateCase test;
    test.matrix.rows = test.matrix.cols = m;
    test.matrix.rowOffsets = {0};
    test.input.resize(m);
    float diagonal = unit ? 1.0f : 2.0f;
    for (int32_t r = 0; r < m; ++r) {
        int32_t logical = lower ? r : m - 1 - r;
        bool wide = logical >= independent;
        for (int32_t c = wide ? 0 : r; c < (wide ? m : r + 1); ++c) {
            int32_t dependency = lower ? c : m - 1 - c;
            Complex value;
            if (c == r) {
                value = Complex(diagonal, 0.0f);
            } else if (wide && dependency < independent) {
                float signedMagnitude = dependency % 2 == 0 ? magnitude : -magnitude;
                value = Complex(signedMagnitude, signedMagnitude);
            } else if (logical == independent + 1 && dependency == independent) {
                value = Complex(0.5f, 0.0f);
            } else {
                continue;
            }
            test.matrix.colIndices.push_back(c);
            test.matrix.values.push_back(value.real());
            test.values.push_back(value);
        }
        test.matrix.rowOffsets.push_back(static_cast<int32_t>(test.values.size()));
        float rhs = diagonal + (logical == independent + 1 ? 0.5f : 0.0f);
        test.input[r] = Complex(rhs, rhs);
    }
    test.matrix.nnz = static_cast<int64_t>(test.values.size());
    return test;
}

void CheckDistributedComplexRows(HandleManager& handle, aclrtStream stream, bool lower, bool unit,
                                 float magnitude)
{
    auto test = MakeDistributedComplexRows(lower, unit, magnitude);
    for (bool inPlace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.lower = lower;
        cfg.unitDiag = unit;
        cfg.inPlace = inPlace;
        auto result = SpSVNpuWrapperComplex(handle.get(), stream, test.matrix, test.values, test.input,
                                            Complex(1.0f, 0.0f), cfg);
        for (size_t r = 0; r < result.y.size(); ++r) {
            SCOPED_TRACE(::testing::Message()
                         << lower << ':' << unit << ':' << magnitude << ':' << inPlace << ':' << r);
            ASSERT_EQ(result.y[r].real(), 1.0f);
            ASSERT_EQ(result.y[r].imag(), 1.0f);
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexDistributedLongRows)
{
    // Two successive singleton levels exercise scratch reuse. Paired complex
    // coefficients cancel exactly, so every component must remain exactly one.
    for (bool lower : {false, true}) {
        for (bool unit : {false, true}) {
            for (float magnitude : {0x1p-6f, 0x1p100f, 0x1p-130f}) {
                ASSERT_NO_FATAL_FAILURE(
                    CheckDistributedComplexRows(handle, env.stream(), lower, unit, magnitude));
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexDeepSingletonWideTransitions)
{
    for (bool lower : {false, true}) {
        constexpr int32_t m = 1400;
        auto test = MakeSingletonWideTransitionCase(lower);
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.lower = lower;
            cfg.unitDiag = false;
            cfg.inPlace = inPlace;
            auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), test.matrix, test.values,
                                                test.input, Complex(1.0f, 0.0f), cfg);
            for (int32_t r = 0; r < m; ++r) {
                SCOPED_TRACE(::testing::Message() << lower << ':' << inPlace << ':' << r);
                ASSERT_NEAR(result.y[r].real(), test.expected[r].real(), 0x1p-18);
                ASSERT_NEAR(result.y[r].imag(), test.expected[r].imag(), 0x1p-18);
            }
        }
    }
}

static void RunFloatCandidateBandwidthCase(HandleManager& handle, aclrtStream stream, int32_t m,
                                           int32_t width)
{
    CsrMatrix matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    for (int32_t row = 0; row < m; ++row) {
        for (int32_t col = std::max(0, row - width); col <= row; ++col) {
            float value = col == row ? 2.f : col == row - 1 ? -0.5f : 0.03125f;
            matrix.colIndices.push_back(col);
            matrix.values.push_back(value);
        }
        matrix.rowOffsets.push_back(static_cast<int32_t>(matrix.values.size()));
    }
    matrix.nnz = static_cast<int64_t>(matrix.values.size());
    for (bool transpose : {false, true}) {
        std::vector<float> rhs(m, 0.f);
        for (int32_t row = 0; row < m; ++row) {
            for (int32_t p = matrix.rowOffsets[row]; p < matrix.rowOffsets[row + 1]; ++p) {
                rhs[transpose ? matrix.colIndices[p] : row] += matrix.values[p];
            }
        }
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.lower = true;
            cfg.inPlace = inPlace;
            cfg.opA = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
            cfg.isRowPtrI64 = m == 128 || m == 129;
            auto result = SpSVNpuWrapper(handle.get(), stream, matrix, rhs, 1.f, cfg);
            for (int32_t row = 0; row < m; ++row) {
                SCOPED_TRACE(::testing::Message()
                             << m << ':' << width << ':' << transpose << ':' << inPlace << ':' << row);
                ASSERT_NEAR(result.y[row], 1.f, 0x1p-18);
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewFloatCandidateBandwidthBoundary)
{
    for (int32_t m : {127, 128, 129, 1024}) {
        for (int32_t width : {1, 12, 13}) {
            RunFloatCandidateBandwidthCase(handle, env.stream(), m, width);
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexScaledExponentAndFallback)
{
    for (bool lower : {false, true}) {
        for (bool overflow : {false, true}) {
            constexpr int32_t m = 1025;
            CsrMatrix matrix;
            matrix.rows = matrix.cols = m;
            matrix.rowOffsets = {0};
            std::vector<Complex> values, input(m);
            for (int32_t row = 0; row < m; ++row) {
                int32_t step = lower ? row : m - 1 - row;
                float coefficient = overflow ? -2.0f : -1.0f;
                for (int32_t col = std::max(0, row - (lower ? 1 : 0)); col <= std::min(m - 1, row + (lower ? 0 : 1));
                     ++col) {
                    float value = col == row ? 1.0f : coefficient;
                    matrix.colIndices.push_back(col);
                    matrix.values.push_back(value);
                    values.emplace_back(value, 0.0f);
                }
                // A tiny nonzero RHS tests a validated-structure rejection
                // in addition to the growing, separately scaled recurrence.
                input[row] = overflow ? (step == 0 ? Complex(1.0f, -1.0f) : Complex(0.0f, 0.0f)) :
                                        Complex(0x1p-130f, -0x1p-130f);
                matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
            }
            matrix.nnz = static_cast<int64_t>(values.size());
            for (bool inPlace : {false, true}) {
                SpSVNpuConfig cfg;
                cfg.lower = lower;
                cfg.inPlace = inPlace;
                auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), matrix, values, input,
                                                    Complex(1.0f, 0.0f), cfg);
                for (int32_t row = 0; row < m; ++row) {
                    int32_t step = lower ? row : m - 1 - row;
                    float expected = overflow ? static_cast<float>(std::ldexp(1.0, step)) :
                                                static_cast<float>(std::ldexp(static_cast<double>(step + 1), -130));
                    SCOPED_TRACE(::testing::Message() << lower << ':' << overflow << ':' << inPlace << ':' << row);
                    EXPECT_EQ(result.y[row].real(), expected);
                    EXPECT_EQ(result.y[row].imag(), -expected);
                }
            }
        }
    }
}

static CertificateCase MakeBinary64NanSuffixFixture(int32_t m)
{
    CertificateCase fixture;
    auto& matrix = fixture.matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    fixture.input.assign(m, Complex(1.0f, 0.5f));
    fixture.expected.resize(m);
    for (int32_t row = 0; row < m; ++row) {
        auto result = std::complex<double>(1.0, 0.5);
        if (row % (m / 2) == 600) {
            matrix.colIndices.push_back(row - 2);
            matrix.values.push_back(4.0f);
            fixture.values.emplace_back(4.0f, 0.0f);
            result -= std::complex<double>(4.0, 0.0) * fixture.expected[row - 2];
        }
        if (row != 0 && row != m / 2) {
            matrix.colIndices.push_back(row - 1);
            matrix.values.push_back(-4.0f);
            fixture.values.emplace_back(-4.0f, 0.0f);
            result -= std::complex<double>(-4.0, 0.0) * fixture.expected[row - 1];
        }
        matrix.colIndices.push_back(row);
        matrix.values.push_back(1.0f);
        fixture.values.emplace_back(1.0f, 0.0f);
        matrix.rowOffsets.push_back(static_cast<int32_t>(fixture.values.size()));
        fixture.expected[row] = result;
    }
    matrix.nnz = static_cast<int64_t>(fixture.values.size());
    return fixture;
}

TEST_F(SpsvCertificateTest, ReviewBinary64NanSuffixKeepsDisconnectedRows)
{
    constexpr int32_t m = 2048;
    auto fixture = MakeBinary64NanSuffixFixture(m);
    ASSERT_TRUE(std::isnan(fixture.expected[m / 2 - 1].real()));
    ASSERT_EQ(fixture.expected[m / 2], std::complex<double>(1.0, 0.5));
    for (bool inPlace : {false, true}) {
        SpSVNpuConfig cfg;
        cfg.unitDiag = true;
        cfg.inPlace = inPlace;
        SCOPED_TRACE(inPlace);
        CheckCertificate(fixture, fixture.expected, cfg);
    }
}

TEST_F(SpsvCertificateTest, ReviewComplexScaledBandwidthBoundary)
{
    for (int32_t bandwidth : {9, 16, 17}) {
        constexpr int32_t m = 1025;
        const Complex expected(2.0f, -1.0f);
        CsrMatrix matrix;
        matrix.rows = matrix.cols = m;
        matrix.rowOffsets = {0};
        std::vector<Complex> values, input(m);
        for (int32_t row = 0; row < m; ++row) {
            Complex sum(0.0f, 0.0f);
            for (int32_t distance : {bandwidth, 1, 0}) {
                if (row < distance) {
                    continue;
                }
                Complex value = distance == 0 ? Complex(1.0f, 0.5f) :
                                distance == 1 ? Complex(-0.5f, 0.125f) :
                                                Complex(0.125f, 0.0f);
                matrix.colIndices.push_back(row - distance);
                matrix.values.push_back(value.real());
                values.push_back(value);
                sum += value;
            }
            input[row] = sum * expected;
            matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
        }
        matrix.nnz = static_cast<int64_t>(values.size());
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.inPlace = inPlace;
            auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), matrix, values, input, Complex(1.0f, 0.0f),
                                                cfg);
            for (int32_t row = 0; row < m; ++row) {
                SCOPED_TRACE(::testing::Message() << bandwidth << ':' << inPlace << ':' << row);
                EXPECT_EQ(result.y[row], expected);
            }
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewRealScaledHasOneComponent)
{
    constexpr int32_t m = 1024;
    for (bool lower : {false, true}) {
        CsrMatrix matrix;
        matrix.rows = matrix.cols = m;
        matrix.rowOffsets = {0};
        std::vector<float> rhs(m, 0.f);
        rhs[lower ? 0 : m - 1] = 1.f;
        for (int32_t row = 0; row < m; ++row) {
            int32_t first = lower ? std::max(0, row - 7) : row;
            int32_t last = lower ? row : std::min(m - 1, row + 7);
            for (int32_t col = first; col <= last; ++col) {
                matrix.colIndices.push_back(col);
                matrix.values.push_back(col == row ? 1.f : std::abs(col - row) == 1 ? -2.f : 0.f);
            }
            matrix.rowOffsets.push_back(static_cast<int32_t>(matrix.values.size()));
        }
        matrix.nnz = static_cast<int64_t>(matrix.values.size());
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.lower = lower;
            cfg.inPlace = inPlace;
            auto result = SpSVNpuWrapper(handle.get(), env.stream(), matrix, rhs, 1.f, cfg);
            for (int32_t row = 0; row < m; ++row) {
                int32_t step = lower ? row : m - 1 - row;
                SCOPED_TRACE(::testing::Message() << lower << ':' << inPlace << ':' << row);
                EXPECT_EQ(result.y[row], static_cast<float>(std::ldexp(1.0, step)));
            }
        }
    }
}

namespace
{

void ExpectReplaySolution(const CsrMatrix& matrix, const std::vector<Complex>& coefficients,
                          const std::vector<Complex>& input, const std::vector<Complex>& actual, bool lower,
                          bool unit = false)
{
    int32_t m = static_cast<int32_t>(input.size());
    std::vector<WideComplex> expected(m);
    for (int32_t step = 0; step < m; ++step) {
        int32_t row = lower ? step : m - 1 - step;
        WideComplex answer(input[row].real(), input[row].imag()), diagonal;
        for (int32_t p = matrix.rowOffsets[row]; p < matrix.rowOffsets[row + 1]; ++p) {
            int32_t col = matrix.colIndices[p];
            WideComplex value(coefficients[p].real(), coefficients[p].imag());
            if (col == row) {
                diagonal = value;
            } else {
                answer -= value * expected[col];
            }
        }
        expected[row] = unit ? answer : answer / diagonal;
        for (int32_t component = 0; component < 2; ++component) {
            float want = static_cast<float>(component == 0 ? expected[row].real() : expected[row].imag());
            float got = component == 0 ? actual[row].real() : actual[row].imag();
            SCOPED_TRACE(row);
            ExpectReplayComponent(want, got);
        }
    }
}
} // namespace

static void RunBinary64ReplayCase(HandleManager& handle, aclrtStream stream, bool lower, bool overflow)
{
    constexpr int32_t m = 16384;
    CsrMatrix matrix;
    matrix.rows = matrix.cols = m;
    matrix.rowOffsets = {0};
    std::vector<Complex> values, input(m);
    for (int32_t row = 0; row < m; ++row) {
        for (int32_t col = lower ? std::max(0, row - 1) : row;
             col <= (lower ? row : std::min(m - 1, row + 1)); ++col) {
            Complex value = col == row ? Complex(0.5f, 0.5f) :
                            overflow ? Complex(-1.f, -1.f) : Complex(-0.125f, 0.0625f);
            matrix.colIndices.push_back(col);
            matrix.values.push_back(value.real());
            values.push_back(value);
        }
        matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
        input[row] = overflow ? Complex(0.f, 0.f) : Complex((row % 7 - 3) * 0.125f, 0.25f);
    }
    matrix.nnz = static_cast<int64_t>(values.size());
    if (overflow) {
        input[lower ? 0 : m - 1] = Complex(1.f, 0.f);
    }
    SpSVNpuConfig cfg;
    cfg.lower = lower;
    cfg.inPlace = !lower;
    cfg.idxBase = lower ? 0 : 1;
    cfg.updateMode = overflow ? "NONE" : "GENERAL";
    auto result = SpSVNpuWrapperComplex(handle.get(), stream, matrix, values, input, Complex(1.f, 0.f), cfg);
    for (int32_t pass = 0; pass < (result.hasSecondSolve ? 2 : 1); ++pass) {
        const auto& coefficients = pass == 0 ? values : result.newValues;
        const auto& actual = pass == 0 ? result.y : result.y2;
        SCOPED_TRACE(::testing::Message() << lower << ':' << overflow << ':' << pass);
        ExpectReplaySolution(matrix, coefficients, input, actual, lower);
    }
}

TEST_F(SpsvCertificateTest, ReviewBinary64ReplayAndRejectedUpdate)
{
    for (bool lower : {false, true}) {
        for (bool overflow : {false, true}) {
            RunBinary64ReplayCase(handle, env.stream(), lower, overflow);
        }
    }
}

TEST_F(SpsvCertificateTest, ReviewUnitReplayNonfiniteComponents)
{
    // Cover one or both nonfinite components in scalar and SIMT replay.
    // UNIT conversion must preserve each component before the next product.
    for (int32_t variant = 0; variant < 4; ++variant) {
        int32_t m = variant < 2 ? 7424 : 16384;
        CsrMatrix matrix;
        matrix.rows = matrix.cols = m;
        matrix.rowOffsets = {0};
        std::vector<Complex> values, input(m);
        input[0] = Complex(variant % 2 ? -1.f : -0.25f, -1.f);
        for (int32_t row = 0; row < m; ++row) {
            if (row != 0) {
                matrix.colIndices.push_back(row - 1);
                Complex coefficient = row == 1025 ? Complex(-0.995664239f, -0.932484627f) : Complex(-2.f, 0.f);
                matrix.values.push_back(coefficient.real());
                values.push_back(coefficient);
            }
            matrix.colIndices.push_back(row);
            matrix.values.push_back(1.f);
            values.emplace_back(1.f, 0.f);
            matrix.rowOffsets.push_back(static_cast<int32_t>(values.size()));
        }
        matrix.nnz = static_cast<int64_t>(values.size());
        for (bool inPlace : {false, true}) {
            SpSVNpuConfig cfg;
            cfg.unitDiag = true;
            cfg.inPlace = inPlace;
            auto result = SpSVNpuWrapperComplex(handle.get(), env.stream(), matrix, values, input,
                                                Complex(1.f, 0.f), cfg);
            SCOPED_TRACE(::testing::Message() << variant << ':' << inPlace);
            EXPECT_EQ(result.y[1024].real(), -std::numeric_limits<float>::infinity());
            EXPECT_EQ(result.y[1024].imag(), -std::numeric_limits<float>::infinity());
            ExpectReplaySolution(matrix, values, input, result.y, true, true);
        }
    }
}

namespace
{
CertificateCase MakeWidePrefixReplayCase(bool lower, bool zeroComponent)
{
    constexpr int32_t m = 1537;
    constexpr int32_t prefix = 512;
    CertificateCase out;
    out.matrix.rows = out.matrix.cols = m;
    out.matrix.rowOffsets = {0};
    out.input.resize(m);
    out.expected.resize(m);
    // The tail exceeds FP32 range while its complete FP64 reference stays
    // finite, requiring replay rather than accepting proposal NaNs.
    Complex coefficient(zeroComponent ? 0.f : -1.25f, zeroComponent ? 1.5f : 0.25f);
    for (int32_t row = 0; row < m; ++row) {
        int32_t logical = lower ? row : m - 1 - row;
        int32_t first = lower && logical >= prefix ? row - 1 : row;
        int32_t last = !lower && logical >= prefix ? row + 1 : row;
        for (int32_t column = first; column <= last; ++column) {
            Complex value = column == row ? Complex(1.f, 0.f) : coefficient;
            out.matrix.colIndices.push_back(column);
            out.matrix.values.push_back(value.real());
            out.values.push_back(value);
        }
        out.matrix.rowOffsets.push_back(static_cast<int32_t>(out.values.size()));
        out.input[row] = Complex(1.f + (row % 7) * 0.125f, 0.5f + (row % 11) * 0.0625f);
    }
    out.matrix.nnz = static_cast<int64_t>(out.values.size());
    for (int32_t logical = 0; logical < m; ++logical) {
        int32_t row = lower ? logical : m - 1 - logical;
        WideComplex sum(out.input[row].real(), out.input[row].imag());
        if (logical >= prefix) {
            sum -= WideComplex(coefficient.real(), coefficient.imag()) * out.expected[lower ? row - 1 : row + 1];
        }
        out.expected[row] = sum;
    }
    return out;
}
} // namespace

TEST_F(SpsvCertificateTest, ReviewBinary64WidePrefixRowOrder)
{
    for (bool lower : {false, true}) {
        for (bool zeroComponent : {false, true}) {
            auto fixture = MakeWidePrefixReplayCase(lower, zeroComponent);
            for (bool inPlace : {false, true}) {
                SpSVNpuConfig cfg;
                cfg.lower = lower;
                cfg.unitDiag = true;
                cfg.inPlace = inPlace;
                cfg.idxBase = lower ? 0 : 1;
                cfg.updateMode = "NONE";
                SCOPED_TRACE(::testing::Message() << lower << ':' << zeroComponent << ':' << inPlace);
                CheckCertificate(fixture, fixture.expected, cfg);
            }
        }
    }
}

}  // namespace sparse_test
