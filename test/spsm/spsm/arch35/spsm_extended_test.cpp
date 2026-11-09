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

#include "test_common.h"
#include "../spsm_golden.h"
#include "../spsm_verify.h"
#include "spsm_npu_wrapper.h"
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <cstring>
#include <numeric>
#include <limits>
#include <random>
#include <chrono>

using namespace sparse_test;
namespace
{
using Complex = std::complex<double>;
constexpr auto SUCCESS = ACL_SPARSE_STATUS_SUCCESS;
struct Entry
{
    int row, col;
    float real, imag;
};

struct Problem
{
    int m = 7, n = 5, format, components, base;
    bool upper, unit, alias, deviceAlpha;
    aclsparseOperation_t opA, opB;
    aclsparseOrder_t orderB, orderC;
    aclDataType type;
    int br, bc, ldb, ldc;
    float alpha[2] { 0.75f, -0.125f };
    std::vector<Entry> entries;
    std::vector<int32_t> ptr, idx;
    std::vector<float> values, input, initialC;
    DeviceBuffer dPtr, dIdx, dValues, dB, dC, dAlpha, workspace;
    SpMatManager a;
    DnMatManager b, c;
    SpSMDescrManager plan;
    const void* alphaPtr = nullptr;

    explicit Problem(int id, int rows = 7, int columns = 5, int pattern = 0)
        : m(rows)
        , n(columns)
        , format(id % 3)
        , components((id / 3) % 2 + 1)
        , base((id / 6) % 2)
        , upper((id / 2) % 2)
        , unit((id / 4) % 2)
        , alias((id / 3) % 2)
        , deviceAlpha((id / 5) % 2)
        , opA(static_cast<aclsparseOperation_t>((id / 6) % 3))
        , opB(static_cast<aclsparseOperation_t>((id / 18) % 3))
        , orderB(static_cast<aclsparseOrder_t>((id / 2) % 2))
        , orderC(static_cast<aclsparseOrder_t>((id / 7) % 2))
        , type(components == 2 ? ACL_COMPLEX64 : ACL_FLOAT)
    {
        BuildSparse(pattern);
        BuildDense();
        dAlpha = DeviceBuffer::copyFrom(alpha, components * sizeof(float));
        alphaPtr = deviceAlpha ? dAlpha.get() : alpha;
    }

    void AppendPatternRow(int r, int pattern)
    {
        entries.push_back({ r, r, 2.0f, 0.0f });
        if (pattern == 4)
        {
            int c = upper ? r + 1 : r - 1;
            if (c >= 0 && c < m)
            {
                entries.push_back({ r, c, 0.0625f, 0.0f });
            }
            return;
        }
        bool fullRow = pattern != 3 && (pattern == 1 || (upper ? r < 3 : r >= m - 3));
        if (fullRow)
        {
            for (int c = 0; c < m; ++c)
            {
                if (upper ? c > r : c < r)
                {
                    entries.push_back({ r, c, pattern == 1 ? -0.3f : 0.0625f, pattern == 1 ? 0.2f : 0.0f });
                }
            }
        }
    }

    void AppendDefaultRow(int r)
    {
        entries.push_back({ r, r, 1.25f, 0.125f });
        entries.push_back({ r, r, 0.75f, 0.25f });
        if (r > 0)
        {
            entries.push_back({ r, r - 1, 0.0625f, -0.03125f });
            entries.push_back({ r, r - 1, 0.03125f, 0.015625f });
            entries.push_back({ r - 1, r, -0.125f, 0.0625f });
        }
    }

    void BuildSparseEntries(int pattern)
    {
        for (int r = 0; r < m; ++r)
        {
            if (pattern == 0)
            {
                AppendDefaultRow(r);
            }
            else
            {
                AppendPatternRow(r, pattern == 5 ? 4 : pattern);
                if (pattern == 5 && r == 0)
                {
                    // Three diagonal slots expose cancellation in a GENERAL update.
                    entries.push_back({ r, r, 0.5f, 0.0f });
                    entries.push_back({ r, r, -0.5f, 0.0f });
                }
            }
        }
        std::reverse(entries.begin(), entries.end());
    }

    void BuildSparseArrays()
    {
        if (format != 2)
        {
            std::stable_sort(entries.begin(), entries.end(), [&](const Entry& x, const Entry& y)
                { return (format == 0 ? x.row : x.col) < (format == 0 ? y.row : y.col); });
            ptr.assign(m + 1, 0);
            for (auto& e : entries)
            {
                ptr[(format == 0 ? e.row : e.col) + 1]++;
            }
            std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
            for (auto& v : ptr)
            {
                v += base;
            }
        }
        for (auto& e : entries)
        {
            if (format == 2)
            {
                ptr.push_back(e.row + base);
            }
            idx.push_back((format == 1 ? e.row : e.col) + base);
            values.push_back(e.real);
            if (components == 2)
            {
                values.push_back(e.imag);
            }
        }
    }

    void CreateSparseDescriptor()
    {
        dPtr = DeviceBuffer::copyFrom(ptr.data(), ptr.size() * 4);
        dIdx = DeviceBuffer::copyFrom(idx.data(), idx.size() * 4);
        dValues = DeviceBuffer::copyFrom(values.data(), values.size() * 4);
        auto indexBase = static_cast<aclsparseIndexBase_t>(base);
        if (format == 0)
        {
            a = SpMatManager::createCsr(m, m, entries.size(), dPtr.get(), dIdx.get(), dValues.get(),
                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, indexBase, type);
        }
        if (format == 1)
        {
            a = SpMatManager::createCsc(m, m, entries.size(), dPtr.get(), dIdx.get(), dValues.get(),
                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, indexBase, type);
        }
        if (format == 2)
        {
            a = SpMatManager::createCoo(
                m, m, entries.size(), dPtr.get(), dIdx.get(), dValues.get(), ACL_SPARSE_INDEX_32I, indexBase, type);
        }
        auto fill = upper ? ACL_SPARSE_FILL_MODE_UPPER : ACL_SPARSE_FILL_MODE_LOWER;
        auto diag = unit ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
        a.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
        a.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    }

    void BuildSparse(int pattern = 0)
    {
        BuildSparseEntries(pattern);
        BuildSparseArrays();
        CreateSparseDescriptor();
    }

    void BuildDense()
    {
        br = opB ? n : m;
        bc = opB ? m : n;
        ldb = (orderB ? br : bc) + 3;
        ldc = (orderC ? m : n) + 5;
        size_t sizeB = static_cast<size_t>(orderB ? bc : br) * ldb * components;
        size_t sizeC = static_cast<size_t>(orderC ? n : m) * ldc * components;
        input.assign(alias ? std::max(sizeB, sizeC) : sizeB, -19.0f);
        for (int r = 0; r < br; ++r)
        {
            for (int col = 0; col < bc; ++col)
            {
                int offset = (orderB ? col * ldb + r : r * ldb + col) * components;
                input[offset] = (r + 1) * 0.25f - col * 0.0625f;
                if (components == 2)
                {
                    input[offset + 1] = r * 0.03125f + col * 0.125f;
                }
            }
        }
        initialC = alias ? input : std::vector<float>(sizeC, -23.0f);
        dB = DeviceBuffer::copyFrom(input.data(), input.size() * 4);
        dC = DeviceBuffer::copyFrom(initialC.data(), initialC.size() * 4);
        b = DnMatManager::create(br, bc, ldb, dB.get(), type, orderB);
        c = DnMatManager::create(m, n, ldc, alias ? dB.get() : dC.get(), type, orderC);
    }

    Eigen::MatrixXcd GoldenMatrix(const std::vector<float>& av, const std::vector<float>* diagonal) const
    {
        Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(m, m);
        for (size_t k = 0; k < entries.size(); ++k)
        {
            const auto& e = entries[k];
            if ((upper && e.col >= e.row) || (!upper && e.col <= e.row))
            {
                matrix(e.row, e.col) += Complex(av[k * components], components == 2 ? av[k * components + 1] : 0.0);
            }
        }
        for (int r = 0; r < m; ++r)
        {
            if (diagonal)
            {
                matrix(r, r)
                    = Complex((*diagonal)[r * components], components == 2 ? (*diagonal)[r * components + 1] : 0.0);
            }
            if (unit)
            {
                matrix(r, r) = 1.0;
            }
        }
        if (opA == ACL_SPARSE_OP_TRANSPOSE)
        {
            matrix = matrix.transpose().eval();
        }
        if (opA == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)
        {
            matrix = matrix.adjoint().eval();
        }
        return matrix;
    }

    Eigen::MatrixXcd GoldenRhs() const
    {
        Eigen::MatrixXcd rhs(br, bc);
        for (int r = 0; r < br; ++r)
        {
            for (int col = 0; col < bc; ++col)
            {
                int offset = (orderB ? col * ldb + r : r * ldb + col) * components;
                rhs(r, col) = Complex(input[offset], components == 2 ? input[offset + 1] : 0.0);
            }
        }
        if (opB == ACL_SPARSE_OP_TRANSPOSE)
        {
            rhs = rhs.transpose().eval();
        }
        if (opB == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE)
        {
            rhs = rhs.adjoint().eval();
        }
        rhs *= Complex(alpha[0], components == 2 ? alpha[1] : 0.0);
        return rhs;
    }

    Eigen::MatrixXcd Golden(const std::vector<float>& av, const std::vector<float>* diagonal = nullptr) const
    {
        Eigen::MatrixXcd matrix = GoldenMatrix(av, diagonal);
        Eigen::MatrixXcd rhs = GoldenRhs();
        if (upper != (opA != ACL_SPARSE_OP_NON_TRANSPOSE))
        {
            return matrix.triangularView<Eigen::Upper>().solve(rhs);
        }
        return matrix.triangularView<Eigen::Lower>().solve(rhs);
    }

    aclsparseStatus_t Size(aclsparseHandle_t h, size_t& bytes)
    {
        return aclsparseSpSMBufferSize(
            h, opA, opB, alphaPtr, a.cget(), b.cget(), c.get(), type, ACL_SPARSE_SPSM_ALG_DEFAULT, plan.get(), &bytes);
    }
    aclsparseStatus_t Analyze(aclsparseHandle_t h)
    {
        return aclsparseSpSMAnalysis(h, opA, opB, alphaPtr, a.cget(), b.cget(), c.get(), type,
            ACL_SPARSE_SPSM_ALG_DEFAULT, plan.get(), workspace.get());
    }
    aclsparseStatus_t Solve(aclsparseHandle_t h)
    {
        return aclsparseSpSM(
            h, opA, opB, alphaPtr, a.cget(), b.cget(), c.get(), type, ACL_SPARSE_SPSM_ALG_DEFAULT, plan.get());
    }
    void RestoreB()
    {
        ASSERT_EQ(
            aclrtMemcpy(dB.get(), dB.size(), input.data(), input.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    }
    std::vector<float> Output()
    {
        std::vector<float> out(initialC.size());
        (alias ? dB : dC).copyToHost(out.data(), out.size() * 4);
        return out;
    }
    void Verify(const Eigen::MatrixXcd& expected)
    {
        auto out = Output();
        std::vector<float> actual[2];
        std::vector<double> golden[2];
        std::vector<bool> used(out.size(), false);
        for (int r = 0; r < m; ++r)
        {
            for (int col = 0; col < n; ++col)
            {
                int offset = (orderC ? col * ldc + r : r * ldc + col) * components;
                actual[0].push_back(out[offset]);
                golden[0].push_back(expected(r, col).real());
                used[offset] = true;
                if (components == 2)
                {
                    actual[1].push_back(out[offset + 1]);
                    golden[1].push_back(expected(r, col).imag());
                    used[offset + 1] = true;
                }
            }
        }
        for (int component = 0; component < components; ++component)
        {
            EXPECT_TRUE(VerifySpsmComponent(
                actual[component], golden[component], component == 0 ? "SpSM950.real" : "SpSM950.imag"));
        }
        for (size_t k = 0; k < out.size(); ++k)
        {
            if (!used[k])
            {
                ASSERT_EQ(std::memcmp(&out[k], &initialC[k], sizeof(float)), 0) << "padding at " << k;
            }
        }
    }
};

template <typename Base>
class SpsmTestEnvironment : public Base
{
public:
    static void SetUpTestSuite()
    {
        env = std::make_unique<AclEnvScope>();
    }
    static void TearDownTestSuite()
    {
        env.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env;
};

class SpsmExtendedTest : public SpsmTestEnvironment<testing::TestWithParam<int>> { };

void SolveAndSynchronize(Problem& p, HandleManager& handle, aclrtStream stream)
{
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

void CheckDeterminism(Problem& p, HandleManager& handle, aclrtStream stream, const std::vector<float>& expected)
{
    p.RestoreB();
    SolveAndSynchronize(p, handle, stream);
    auto repeat = p.Output();
    ASSERT_EQ(expected.size(), repeat.size());
    EXPECT_EQ(std::memcmp(expected.data(), repeat.data(), expected.size() * sizeof(float)), 0);
}

void CheckGeneralUpdate(Problem& p, HandleManager& handle, aclrtStream stream, std::vector<float>& updated)
{
    for (auto& value : updated)
    {
        value *= 1.25f;
    }
    auto updateBuffer = DeviceBuffer::copyFrom(updated.data(), updated.size() * sizeof(float));
    ASSERT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), updateBuffer.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        SUCCESS);
    p.RestoreB();
    SolveAndSynchronize(p, handle, stream);
    p.Verify(p.Golden(updated));
    auto first = p.Output();
    ASSERT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), updateBuffer.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        SUCCESS);
    CheckDeterminism(p, handle, stream, first);
}

void CheckDiagonalUpdate(Problem& p, HandleManager& handle, aclrtStream stream, const std::vector<float>& values)
{
    std::vector<float> diagonal(p.m * p.components, 0.125f);
    for (int r = 0; r < p.m; ++r)
    {
        diagonal[r * p.components] = 3.0f + r * 0.125f;
    }
    auto diagonalBuffer = DeviceBuffer::copyFrom(diagonal.data(), diagonal.size() * sizeof(float));
    ASSERT_EQ(
        aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), diagonalBuffer.get(), ACL_SPARSE_SPSM_UPDATE_DIAGONAL),
        SUCCESS);
    p.RestoreB();
    SolveAndSynchronize(p, handle, stream);
    p.Verify(p.Golden(values, &diagonal));
    if (!p.unit)
    {
        auto bad = diagonal;
        std::fill_n(bad.begin(), p.components, 0.0f);
        auto badBuffer = DeviceBuffer::copyFrom(bad.data(), bad.size() * sizeof(float));
        EXPECT_EQ(aclsparseSpSMUpdateMatrix(
                      handle.get(), p.plan.get(), badBuffer.get(), ACL_SPARSE_SPSM_UPDATE_DIAGONAL),
            ACL_SPARSE_STATUS_NOT_SUPPORTED);
        p.RestoreB();
        SolveAndSynchronize(p, handle, stream);
        p.Verify(p.Golden(values, &diagonal));
    }
}

TEST_P(SpsmExtendedTest, FormatsOperationsUpdatesAndDeterminism)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(GetParam());
    ASSERT_EQ(aclsparseSetPointerMode(
                  handle.get(), p.deviceAlpha ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        SUCCESS);
    size_t bytes = 0;
    // The public setter supports delayed dense value binding.
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), nullptr), SUCCESS);
    ASSERT_EQ(aclsparseDnMatSetValues(p.c.get(), nullptr), SUCCESS);
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), p.dB.get()), SUCCESS);
    ASSERT_EQ(aclsparseDnMatSetValues(p.c.get(), p.alias ? p.dB.get() : p.dC.get()), SUCCESS);
    SolveAndSynchronize(p, handle, env->stream());
    p.Verify(p.Golden(p.values));
    auto first = p.Output();
    CheckDeterminism(p, handle, env->stream(), first);
    p.RestoreB();
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    SolveAndSynchronize(p, handle, env->stream());
    auto repeat = p.Output();
    EXPECT_EQ(std::memcmp(first.data(), repeat.data(), first.size() * sizeof(float)), 0);
    auto updated = p.values;
    CheckGeneralUpdate(p, handle, env->stream(), updated);
    CheckDiagonalUpdate(p, handle, env->stream(), updated);
}

float MixedValue(int index, std::mt19937& rng)
{
    // Every complete group of 20 contains exactly 14 uniform, 4 normal and
    // 2 constructed values. The diagonal-only system needs no conditioning.
    int kind = index % 20;
    if (kind < 14)
    {
        return std::uniform_real_distribution<float>(-1, 1)(rng);
    }
    if (kind < 18)
    {
        return std::normal_distribution<float>(0, 1)(rng);
    }
    return kind == 18 ? 1.0f : -1.0f;
}

TEST_P(SpsmExtendedTest, MixedDistributionSeventyTwentyTen)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(GetParam(), 20, 10, 3);
    ASSERT_EQ(aclsparseSetPointerMode(
                  handle.get(), p.deviceAlpha ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        SUCCESS);
    std::mt19937 rng(20260903 + GetParam());
    for (size_t i = 0; i < p.entries.size(); ++i)
    {
        for (int c = 0; c < p.components; ++c)
        {
            p.values[i * p.components + c] = MixedValue(i, rng);
        }
    }
    for (int r = 0, i = 0; r < p.br; ++r)
    {
        for (int col = 0; col < p.bc; ++col, ++i)
        {
            int offset = (p.orderB ? col * p.ldb + r : r * p.ldb + col) * p.components;
            for (int c = 0; c < p.components; ++c)
            {
                p.input[offset + c] = MixedValue(i, rng);
            }
        }
    }
    if (p.alias)
    {
        p.initialC = p.input;
    }
    ASSERT_EQ(
        aclrtMemcpy(p.dValues.get(), p.dValues.size(), p.values.data(), p.values.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    p.RestoreB();
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
    std::cout << "[distribution] A=14/4/2 B=140/40/20 (uniform/normal/constructed) per component\n";
}

INSTANTIATE_TEST_SUITE_P(Ascend950, SpsmExtendedTest, testing::Range(0, 54));

class SpsmBoundaryTest : public SpsmTestEnvironment<testing::Test> { };

TEST_F(SpsmBoundaryTest, StageAndPointerModeValidation)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(p.Analyze(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    auto wrongFill = ACL_SPARSE_FILL_MODE_UPPER;
    p.a.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &wrongFill, sizeof(wrongFill));
    EXPECT_EQ(p.Analyze(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    auto fill = ACL_SPARSE_FILL_MODE_LOWER;
    p.a.setAttribute(ACL_SPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_DEVICE), SUCCESS);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_HOST), SUCCESS);
    auto otherB = DeviceBuffer::alloc(p.dB.size());
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), otherB.get()), SUCCESS);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), p.dB.get()), SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, InvalidIndicesAndFailedGeneralUpdate)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(2); // COO, upper, non-unit
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    auto badIdx = p.idx;
    badIdx[0] = -1;
    ASSERT_EQ(aclrtMemcpy(p.dIdx.get(), p.dIdx.size(), badIdx.data(), badIdx.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    EXPECT_EQ(p.Analyze(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclrtMemcpy(p.dIdx.get(), p.dIdx.size(), p.idx.data(), p.idx.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    auto badValues = p.values;
    for (size_t k = 0; k < p.entries.size(); ++k)
    {
        if (p.entries[k].row == 0 && p.entries[k].col == 0)
        {
            badValues[k] = 0.0f;
        }
    }
    auto bad = DeviceBuffer::copyFrom(badValues.data(), badValues.size() * 4);
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), bad.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, CrossCoreDependenciesShareCacheLines)
{
    HandleManager handle;
    handle.setStream(env->stream());
    // Seven FP32 RHSs make adjacent rows share a 128-byte cache line. A^T
    // turns three long rows into four levels with different row owners.
    Problem p(24, 31, 7, 2);
    ASSERT_FALSE(p.unit);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        p.Verify(p.Golden(p.values));
    }
}

TEST_F(SpsmBoundaryTest, ComplexUnitRoundingPropagation)
{
    HandleManager handle;
    handle.setStream(env->stream());
    // Dense unit triangular dependencies amplify FP32 intermediate rounding;
    // this uses a different size and fixture from the external test package.
    Problem p(15, 97, 9, 1);
    ASSERT_EQ(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_DEVICE), SUCCESS);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, RealCancellationRetainsProductAndDivisionTails)
{
    // A near-cancelling product leaves -2^-46 in row 1. Row 2 magnifies
    // it into a visible result. Check both level-parallel and serial graphs,
    // with an explicit division as well as an implicit unit diagonal.
    for (int rows : { 3, 130 })
    {
        for (bool unit : { false, true })
        {
            SCOPED_TRACE(rows);
            SCOPED_TRACE(unit);
            HandleManager handle;
            handle.setStream(env->stream());
            Problem p(0, rows, 1, 4);
            p.unit = unit;
            auto diag = unit ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
            p.a.setAttribute(ACL_SPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
            p.alpha[0] = 1.0f;
            for (size_t k = 0; k < p.entries.size(); ++k)
            {
                const auto& e = p.entries[k];
                p.values[k] = e.row == e.col ? (e.row == 1 ? 3.0f : 1.0f)
                                             : (e.row == 1 ? 1.0f + 0x1p-23f : (e.row == 2 ? 0x1p24f : 1.0f));
            }
            for (int r = 0; r < rows; ++r)
            {
                p.input[r * p.ldb] = r == 0 ? 1.0f + 0x1p-23f : (r == 1 ? 1.0f + 0x1p-22f : 0.0f);
            }
            p.RestoreB();
            ASSERT_EQ(aclrtMemcpy(p.dValues.get(), p.dValues.size(), p.values.data(), p.values.size() * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_SUCCESS);
            size_t bytes = 0;
            ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
            p.workspace = DeviceBuffer::alloc(bytes);
            ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
            ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
            auto output = p.Output();
            double expected = 0x1p-22 / (unit ? 1.0 : 3.0);
            for (int r = 2; r < rows; ++r)
            {
                EXPECT_NEAR(output[r * p.ldc], expected, 1e-12) << "row " << r;
                expected = -expected;
            }
            p.Verify(p.Golden(p.values));
        }
    }
}

TEST_F(SpsmBoundaryTest, RhsWorkspacesPreserveTransposedAliasedInput)
{
    for (bool window : { false, true })
    {
        SCOPED_TRACE(window);
        HandleManager handle;
        handle.setStream(env->stream());
        // Exceed the tail budget: use a partial last panel or a short row window.
        // B is transposed and overlaps C, so publishing a panel early would
        // corrupt the original inputs required by a later panel.
        Problem p(0, window ? 2053 : 257, window ? 2049 : 16321, window ? 4 : 2);
        p.opB = ACL_SPARSE_OP_TRANSPOSE;
        p.alias = true;
        p.alpha[0] = 1.0f;
        p.BuildDense();
        for (int r = 0; r < p.br; ++r)
        {
            for (int c = 0; c < p.bc; ++c)
            {
                p.input[r * p.ldb + c] = window ? (c == 0 ? 2.0f : 2.0625f) : (c < p.m - 3 ? 2.0f : 2.0f + 0.0625f * c);
            }
        }
        p.initialC = p.input;
        size_t bytes = 0;
        ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
        p.workspace = DeviceBuffer::alloc(bytes);
        ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            p.RestoreB();
            ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
            auto output = p.Output();
            for (int r = 0; r < p.m; ++r)
            {
                for (int c = 0; c < p.n; ++c)
                {
                    ASSERT_FLOAT_EQ(output[r * p.ldc + c], 1.0f) << "row " << r << ", rhs " << c;
                }
            }
        }
    }
}

TEST_F(SpsmBoundaryTest, FailedUpdateBeyondFirstThreadTilePreservesPlan)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(24, 513, 3, 2);
    ASSERT_FALSE(p.unit);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    auto badValues = p.values;
    for (size_t k = 0; k < p.entries.size(); ++k)
    {
        if (p.entries[k].row == 300 && p.entries[k].col == 300)
        {
            badValues[k] = 0;
        }
    }
    auto bad = DeviceBuffer::copyFrom(badValues.data(), badValues.size() * sizeof(float));
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), bad.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, GeneralUpdateCompensatesDuplicateDiagonal)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0, 7, 2, 5);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);

    auto updated = p.values;
    constexpr float diagonal[] = { 16777216.0f, 1.0f, -16777216.0f };
    int slots = 0;
    for (size_t k = 0; k < p.entries.size(); ++k)
    {
        if (p.entries[k].row == 0 && p.entries[k].col == 0)
        {
            ASSERT_LT(slots, 3);
            updated[k] = diagonal[slots++];
        }
    }
    ASSERT_EQ(slots, 3);
    auto updateBuffer = DeviceBuffer::copyFrom(updated.data(), updated.size() * sizeof(float));
    ASSERT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), updateBuffer.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        SUCCESS);
    SolveAndSynchronize(p, handle, env->stream());
    p.Verify(p.Golden(updated));

    auto zeroDiagonal = updated;
    for (size_t k = 0; k < p.entries.size(); ++k)
    {
        if (p.entries[k].row == 0 && p.entries[k].col == 0)
        {
            zeroDiagonal[k] = 0.0f;
        }
    }
    auto invalidBuffer = DeviceBuffer::copyFrom(zeroDiagonal.data(), zeroDiagonal.size() * sizeof(float));
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), invalidBuffer.get(),
                  ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_NOT_SUPPORTED);
    p.RestoreB();
    SolveAndSynchronize(p, handle, env->stream());
    p.Verify(p.Golden(updated));
}

TEST_F(SpsmBoundaryTest, NullAnalysisBufferChecksHandleWorkspaceCapacity)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    auto shortWorkspace = DeviceBuffer::alloc(64);
    ASSERT_GT(bytes, shortWorkspace.size());
    ASSERT_EQ(aclsparseSetWorkspace(handle.get(), shortWorkspace.get(), shortWorkspace.size()), SUCCESS);
    EXPECT_EQ(p.Analyze(handle.get()), ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES);
    ASSERT_EQ(aclsparseSetWorkspace(handle.get(), nullptr, 0), SUCCESS);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    SolveAndSynchronize(p, handle, env->stream());
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, WorkspaceAndUpdateAllocationBounds)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    EXPECT_EQ(aclsparseSpSMAnalysis(handle.get(), p.opA, p.opB, p.alphaPtr, p.a.cget(), p.b.cget(), p.c.get(), p.type,
                  ACL_SPARSE_SPSM_ALG_DEFAULT, p.plan.get(), nullptr),
        SUCCESS);
    auto shortBuffer = DeviceBuffer::alloc(64);
    EXPECT_EQ(aclsparseSpSMAnalysis(handle.get(), p.opA, p.opB, p.alphaPtr, p.a.cget(), p.b.cget(), p.c.get(), p.type,
                  ACL_SPARSE_SPSM_ALG_DEFAULT, p.plan.get(), shortBuffer.get()),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    p.workspace = DeviceBuffer::alloc(bytes);
    auto misaligned = static_cast<char*>(p.workspace.get()) + 1;
    EXPECT_EQ(aclsparseSpSMAnalysis(handle.get(), p.opA, p.opB, p.alphaPtr, p.a.cget(), p.b.cget(), p.c.get(), p.type,
                  ACL_SPARSE_SPSM_ALG_DEFAULT, p.plan.get(), misaligned),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    auto shortValues = DeviceBuffer::alloc(4);
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), shortValues.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), nullptr, ACL_SPARSE_SPSM_UPDATE_DIAGONAL),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), p.values.data(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), p.dValues.get(), static_cast<aclsparseSpSMUpdate_t>(99)),
        ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
    // No work is pending here. Free before a new Solve to test rejection
    // without ever freeing storage while a kernel is using it.
    p.workspace = DeviceBuffer();
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpSMUpdateMatrix(handle.get(), p.plan.get(), p.dValues.get(), ACL_SPARSE_SPSM_UPDATE_GENERAL),
        ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpsmBoundaryTest, HostPointersAndDestroyedPlanRejected)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_DEVICE), SUCCESS);
    EXPECT_EQ(p.Size(handle.get(), bytes), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_HOST), SUCCESS);
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), nullptr), SUCCESS);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), p.input.data()), SUCCESS);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclsparseDnMatSetValues(p.b.get(), p.dB.get()), SUCCESS);
    // release() 先取走句柄：显式销毁与二次销毁均由本用例接管，
    // 避免 ~SpSMDescrManager 对已销毁句柄再次 DestroyDescr（UAF/double free）。
    aclsparseSpSMDescr_t rawPlan = p.plan.release();
    ASSERT_EQ(aclsparseSpSMDestroyDescr(rawPlan), SUCCESS);
    EXPECT_EQ(p.Solve(handle.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseSpSMDestroyDescr(rawPlan), ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpsmBoundaryTest, LargeFiniteComplexDivision)
{
    // Test numerator overflow alone, denominator overflow, and both Smith
    // branches. The reference retains FP64 throughout the triangular solve.
    for (auto diagonal : { Complex(2.0, 2.0), Complex(3e38, 3e38), Complex(2e38, 3e38) })
    {
        SCOPED_TRACE(diagonal);
        HandleManager handle;
        handle.setStream(env->stream());
        Problem p(3, 1, 2, 3);
        p.alpha[0] = 1.0f;
        p.alpha[1] = 0.0f;
        p.values[0] = static_cast<float>(diagonal.real());
        p.values[1] = static_cast<float>(diagonal.imag());
        for (int col = 0; col < 2; ++col)
        {
            int offset = (p.orderB ? col * p.ldb : col) * p.components;
            p.input[offset] = 3e38f;
            p.input[offset + 1] = col == 0 ? 3e38f : -3e38f;
        }
        p.initialC = p.input;
        p.RestoreB();
        ASSERT_EQ(aclrtMemcpy(p.dValues.get(), p.dValues.size(), p.values.data(), p.values.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        size_t bytes = 0;
        ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
        p.workspace = DeviceBuffer::alloc(bytes);
        ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
        ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        p.Verify(p.Golden(p.values));
    }
}

void SetNonfiniteInput(Problem& p, const float* special)
{
    for (int col = 0; col < 4; ++col)
    {
        int offset = (p.orderB ? col * p.ldb : col) * p.components;
        p.input[offset] = special[col];
        if (p.components == 2)
        {
            p.input[offset + 1] = 0.5f;
        }
    }
    if (p.unit)
    {
        p.values[0] = std::numeric_limits<float>::quiet_NaN();
    }
    if (p.alias)
    {
        p.initialC = p.input;
    }
}

Eigen::MatrixXcd NonfiniteGolden(const Problem& p, const float* special)
{
    Eigen::MatrixXcd golden(1, 4);
    for (int col = 0; col < 4; ++col)
    {
        if (p.components == 1)
        {
            golden(0, col) = Complex(double(p.alpha[0]) * special[col] / (p.unit ? 1 : 2), 0);
        }
        else
        {
            Complex x(special[col], 0.5), alpha(p.alpha[0], p.alpha[1]);
            golden(0, col) = (alpha * x) / (p.unit ? 1.0 : 2.0);
        }
    }
    return golden;
}

void RunNonfiniteCase(int id, aclrtStream stream)
{
    HandleManager handle;
    handle.setStream(stream);
    Problem p(id, 1, 4, 3);
    ASSERT_EQ(aclsparseSetPointerMode(
                  handle.get(), p.deviceAlpha ? ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        SUCCESS);
    float special[] = { std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), 0.0f };
    SetNonfiniteInput(p, special);
    p.RestoreB();
    ASSERT_EQ(aclrtMemcpy(p.dValues.get(), p.dValues.size(), p.values.data(), p.values.size() * sizeof(float),
                  ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    SolveAndSynchronize(p, handle, stream);
    p.Verify(NonfiniteGolden(p, special));
}

TEST_F(SpsmBoundaryTest, NonfiniteRhsAndUnitDiagonal)
{
    for (int id : { 0, 3, 4, 8 })
    {
        SCOPED_TRACE(id);
        RunNonfiniteCase(id, env->stream());
    }
}

TEST_F(SpsmBoundaryTest, CancellationUsesHighPrecisionGolden)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0, 31, 7, 4);
    // B is formed from a known alternating-sign solution. Neighbor terms
    // almost cancel while all comparisons retain the actual FP32 inputs.
    for (size_t i = 0; i < p.entries.size(); ++i)
    {
        p.values[i] = p.entries[i].row == p.entries[i].col ? 1.0f : .999f;
    }
    for (int r = 0; r < p.br; ++r)
    {
        for (int c = 0; c < p.bc; ++c)
        {
            p.input[r * p.ldb + c] = (r % 2 ? -1.0f : 1.0f) * (r ? .001f : 1.0f) * (c + 1);
        }
    }
    p.RestoreB();
    ASSERT_EQ(
        aclrtMemcpy(p.dValues.get(), p.dValues.size(), p.values.data(), p.values.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    p.Verify(p.Golden(p.values));
}

TEST_F(SpsmBoundaryTest, IndependentStreamsAndStreamMutation)
{
    aclrtStream other = nullptr;
    ASSERT_EQ(aclrtCreateStream(&other), ACL_SUCCESS);
    {
        HandleManager first, second;
        first.setStream(env->stream());
        second.setStream(other);
        Problem p(0), q(1);
        size_t bytes = 0;
        ASSERT_EQ(p.Size(first.get(), bytes), SUCCESS);
        p.workspace = DeviceBuffer::alloc(bytes);
        ASSERT_EQ(q.Size(second.get(), bytes), SUCCESS);
        q.workspace = DeviceBuffer::alloc(bytes);
        ASSERT_EQ(p.Analyze(first.get()), SUCCESS);
        ASSERT_EQ(q.Analyze(second.get()), SUCCESS);
        first.setStream(other);
        EXPECT_EQ(p.Solve(first.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
        first.setStream(env->stream());
        ASSERT_EQ(p.Solve(first.get()), SUCCESS);
        ASSERT_EQ(q.Solve(second.get()), SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(other), ACL_SUCCESS);
        p.Verify(p.Golden(p.values));
        q.Verify(q.Golden(q.values));
    }
    ASSERT_EQ(aclrtDestroyStream(other), ACL_SUCCESS);
}

TEST_F(SpsmBoundaryTest, SolveReturnsBeforeDeviceCompletion)
{
    HandleManager handle;
    handle.setStream(env->stream());
    Problem p(0, 65537, 31, 4);
    size_t bytes = 0;
    ASSERT_EQ(p.Size(handle.get(), bytes), SUCCESS);
    p.workspace = DeviceBuffer::alloc(bytes);
    ASSERT_EQ(p.Analyze(handle.get()), SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    aclrtEvent finished = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&finished), ACL_SUCCESS);
    ASSERT_EQ(p.Solve(handle.get()), SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(finished, env->stream()), ACL_SUCCESS);
    aclrtEventRecordedStatus state;
    ASSERT_EQ(aclrtQueryEventStatus(finished, &state), ACL_SUCCESS);
    EXPECT_EQ(state, ACL_EVENT_RECORDED_STATUS_NOT_READY);
    ASSERT_EQ(aclrtSynchronizeEvent(finished), ACL_SUCCESS);
    ASSERT_EQ(aclrtQueryEventStatus(finished, &state), ACL_SUCCESS);
    EXPECT_EQ(state, ACL_EVENT_RECORDED_STATUS_COMPLETE);
    ASSERT_EQ(aclrtDestroyEvent(finished), ACL_SUCCESS);
    // Releasing storage is safe now, after the recorded completion event.
}

TEST(SpsmPrecisionRules, RejectOutliersAndWrongSpecialValues)
{
    std::vector<float> x(100, 1.0f);
    std::vector<double> y(100, 1.0);
    x[0] = 1.005f;
    EXPECT_TRUE(VerifySpsmComponent(x, y, "99percent-within-absolute-limit"));
    x[0] = 1.02f;
    EXPECT_FALSE(VerifySpsmComponent(x, y, "99percent-outlier-must-fail"));
    EXPECT_FALSE(VerifySpsmComponent(
        { std::numeric_limits<float>::infinity() }, { -std::numeric_limits<double>::infinity() }, "opposite-inf"));
    EXPECT_TRUE(VerifySpsmComponent(
        { std::numeric_limits<float>::quiet_NaN() }, { std::numeric_limits<double>::quiet_NaN() }, "matching-nan"));
}

// Sparse FP64 Eigen golden keeps these tests O(m + nnz), including the long row.
void RunLargeBoundary(HandleManager& handle, aclrtStream stream, int m, bool chain)
{
    constexpr int n = 3;
    std::vector<int32_t> row(m + 1, 0), col;
    std::vector<float> values, rhs(m * n, 1.0f);
    for (int r = 0; r < m; ++r)
    {
        int start = chain ? std::max(0, r - 1) : (r == m - 1 ? 0 : r);
        for (int c = start; c <= r; ++c)
        {
            col.push_back(c);
            values.push_back(c == r ? 2.0f : 0.0000152587890625f);
        }
        row[r + 1] = col.size();
    }
    auto golden = SpsmGolden(row, col, values, m, n, n, rhs, 1.0f, true, false, false, true);
    auto result = SpsmNpu(handle, stream, row, col, values, values.size(), rhs, m, n, n, 1.0f,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_FILL_MODE_LOWER,
        ACL_SPARSE_DIAG_TYPE_NON_UNIT, ACL_SPARSE_ORDER_ROW, false, ACL_SPARSE_INDEX_BASE_ZERO);
    ASSERT_EQ(result.bufferSizeRet, SUCCESS);
    ASSERT_EQ(result.analysisRet, SUCCESS);
    ASSERT_EQ(result.solveRet, SUCCESS);
    EXPECT_TRUE(VerifySpsmComponent(result.X, golden.X, chain ? "DeepChain" : "LongRow"));
}

TEST_F(SpsmBoundaryTest, RowLargerThanUb)
{
    HandleManager handle;
    handle.setStream(env->stream());
    RunLargeBoundary(handle, env->stream(), 32769, false);
}

TEST_F(SpsmBoundaryTest, LevelTableLargerThanUb)
{
    HandleManager handle;
    handle.setStream(env->stream());
    RunLargeBoundary(handle, env->stream(), 65537, true);
}
} // namespace
