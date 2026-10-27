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

#include "sddmm_test_support.h"

#include <algorithm>

namespace sparse_test {
namespace sddmm_extended {
namespace {

constexpr int64_t kWorkspaceTestRows = 16;
constexpr int64_t kWorkspaceTestNnzPerRow = 64;
#ifdef SDDMM_TEST_WITH_SPMM
constexpr int64_t kWorkspaceTestSpmmColumns = 8;
#endif

// Each row of X and each row of sparse values is constant. Y is 1/8, so
// SDDMM's expected value is k*(row+1)/128 exactly in FP32.
class WarmWorkspaceFixture {
public:
    WarmWorkspaceFixture(aclrtStream stream, bool cube)
        : n(cube ? 320 : 128), k(cube ? 128 : 32)
    {
        std::vector<int32_t> offsets(kWorkspaceTestRows + 1);
        std::vector<int32_t> columns(kWorkspaceTestRows * kWorkspaceTestNnzPerRow);
        std::vector<float> x(kWorkspaceTestRows * k);
        std::vector<float> values(columns.size());
        for (int64_t row = 0; row < kWorkspaceTestRows; ++row) {
            offsets[row] = static_cast<int32_t>(row * kWorkspaceTestNnzPerRow);
            auto first = columns.begin() + offsets[row];
            for (int64_t slot = 0; slot < kWorkspaceTestNnzPerRow; ++slot) {
                first[slot] = static_cast<int32_t>((row * 17 + slot) % n);
            }
            std::sort(first, first + kWorkspaceTestNnzPerRow);
            const float rowValue = static_cast<float>(row + 1) / 16.0F;
            std::fill_n(x.begin() + row * k, k, rowValue);
            std::fill_n(values.begin() + offsets[row], kWorkspaceTestNnzPerRow, rowValue);
        }
        offsets.back() = static_cast<int32_t>(columns.size());
        const std::vector<float> y(k * n, 0.125F);
        dOffsets = DeviceBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
        dColumns = DeviceBuffer::copyFrom(columns.data(), columns.size() * sizeof(int32_t));
        dValues = DeviceBuffer::copyFrom(values.data(), values.size() * sizeof(float));
        dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
        dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));
        matC = SpMatManager::createCsr(kWorkspaceTestRows, n, columns.size(),
            dOffsets.get(), dColumns.get(), dValues.get());
        matX = DnMatManager::create(kWorkspaceTestRows, k, k,
            dX.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matY = DnMatManager::create(k, n, n, dY.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        SetHostTestStream(handle, stream);
        CheckSetupStatus(aclsparseSDDMMBufferSize(handle.get(),
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha, matX.cget(), matY.cget(), &beta, matC.get(), ACL_FLOAT,
            ACL_SPARSE_SDDMM_ALG_DEFAULT, &workspaceBytes), "aclsparseSDDMMBufferSize");
    }

    aclsparseStatus_t Prepare(void *workspace)
    {
        return aclsparseSDDMMPreprocess(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matX.cget(), matY.cget(),
            &beta, matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace);
    }

    aclsparseStatus_t Execute(void *workspace)
    {
        return aclsparseSDDMM(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matX.cget(), matY.cget(),
            &beta, matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace);
    }

    void CheckValues() const
    {
        RealRunResult result;
        result.actual.resize(kWorkspaceTestRows * kWorkspaceTestNnzPerRow);
        result.golden.resize(result.actual.size());
        dValues.copyToHost(result.actual.data(), result.actual.size() * sizeof(float));
        for (int64_t row = 0; row < kWorkspaceTestRows; ++row) {
            const float value = static_cast<float>(k * (row + 1)) / 128.0F;
            std::fill_n(result.golden.begin() + row * kWorkspaceTestNnzPerRow,
                kWorkspaceTestNnzPerRow, value);
        }
        EXPECT_TRUE(VerifyRealRun(result, ACL_FLOAT, "SDDMM_WARM_WORKSPACE"));
    }

    int64_t n;
    int64_t k;
    float alpha = 1.0F;
    float beta = 0.0F;
    size_t workspaceBytes = 0;
    DeviceBuffer dOffsets;
    DeviceBuffer dColumns;
    DeviceBuffer dValues;
    DeviceBuffer dX;
    DeviceBuffer dY;
    SpMatManager matC;
    DnMatManager matX;
    DnMatManager matY;
    HandleManager handle;
};

#ifdef SDDMM_TEST_WITH_SPMM
class SpmmWorkspaceFixture {
public:
    explicit SpmmWorkspaceFixture(WarmWorkspaceFixture &sddmm) : sddmm_(sddmm)
    {
        std::vector<float> b(sddmm.n * kWorkspaceTestSpmmColumns);
        for (int64_t row = 0; row < sddmm.n; ++row) {
            for (int64_t column = 0; column < kWorkspaceTestSpmmColumns; ++column) {
                b[row * kWorkspaceTestSpmmColumns + column] =
                    static_cast<float>(column + 1) / 16.0F;
            }
        }
        const std::vector<float> c(kWorkspaceTestRows * kWorkspaceTestSpmmColumns, 0.0F);
        dB_ = DeviceBuffer::copyFrom(b.data(), b.size() * sizeof(float));
        dC_ = DeviceBuffer::copyFrom(c.data(), c.size() * sizeof(float));
        matB_ = DnMatManager::create(sddmm.n, kWorkspaceTestSpmmColumns,
            kWorkspaceTestSpmmColumns, dB_.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matC_ = DnMatManager::create(kWorkspaceTestRows, kWorkspaceTestSpmmColumns,
            kWorkspaceTestSpmmColumns, dC_.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        CheckSetupStatus(aclsparseSpMMGetBufferSize(sddmm.handle.get(),
            ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, sddmm.matC.cget(), matB_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, &workspaceBytes), "aclsparseSpMMGetBufferSize");
    }

    aclsparseStatus_t Prepare(void *workspace)
    {
        return aclsparseSpMMPreprocess(sddmm_.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_, sddmm_.matC.cget(), matB_.cget(),
            &beta_, matC_.get(), ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, workspace);
    }

    aclsparseStatus_t Execute(void *workspace)
    {
        return aclsparseSpMM(sddmm_.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_, sddmm_.matC.cget(), matB_.cget(),
            &beta_, matC_.get(), ACL_FLOAT, ACL_SPARSE_SPMM_ALG_DEFAULT, workspace);
    }

    void CheckValues(bool afterSddmm) const
    {
        RealRunResult result;
        result.actual.resize(kWorkspaceTestRows * kWorkspaceTestSpmmColumns);
        result.golden.resize(result.actual.size());
        dC_.copyToHost(result.actual.data(), result.actual.size() * sizeof(float));
        // Each output sums 64 identical terms. Both factors are binary
        // fractions, giving an exact analytical reference for this fixture.
        const float scale = afterSddmm ? static_cast<float>(sddmm_.k) / 32.0F : 0.25F;
        for (int64_t row = 0; row < kWorkspaceTestRows; ++row) {
            for (int64_t column = 0; column < kWorkspaceTestSpmmColumns; ++column) {
                result.golden[row * kWorkspaceTestSpmmColumns + column] =
                    scale * static_cast<float>((row + 1) * (column + 1));
            }
        }
        EXPECT_TRUE(VerifyRealRun(result, ACL_FLOAT, "SPMM_AFTER_WARM_SDDMM"));
    }

    size_t workspaceBytes = 0;

private:
    WarmWorkspaceFixture &sddmm_;
    float alpha_ = 1.0F;
    float beta_ = 0.0F;
    DeviceBuffer dB_;
    DeviceBuffer dC_;
    DnMatManager matB_;
    DnMatManager matC_;
};
#endif

void CheckWarmPreparedState(const WarmWorkspaceFixture &fixture, uint64_t epoch)
{
    const auto &state = fixture.matC.cget()->sddmmPrepared;
    EXPECT_TRUE(state.prepared);
    EXPECT_EQ(state.preprocessEpoch, epoch);
    if (fixture.k == 128) {
        EXPECT_NE(state.patternFlags & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
        EXPECT_EQ(state.patternLogicalFirstColumn, 0);
    }
}

} // namespace

void CheckWarmWorkspaceOwners(aclrtStream stream, bool cube)
{
    WarmWorkspaceFixture fixture(stream, cube);
    ASSERT_GT(fixture.workspaceBytes, static_cast<size_t>(kSddmmWorkspaceBytes));
    DeviceBuffer workspace = DeviceBuffer::alloc(fixture.workspaceBytes);
    DeviceBuffer distinct = DeviceBuffer::alloc(fixture.workspaceBytes);
    ASSERT_EQ(fixture.Prepare(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    auto *descr = fixture.matC.get();
    const uint64_t epoch = descr->sddmmPrepared.preprocessEpoch;
    descr->activeBuffer = distinct.get();
    descr->activeSpmvBuffer = distinct.get();
    ASSERT_EQ(fixture.Execute(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(descr->activeBuffer, distinct.get());
    EXPECT_EQ(descr->activeSpmvBuffer, distinct.get());
    CheckWarmPreparedState(fixture, epoch);
    fixture.CheckValues();
    descr->activeBuffer = workspace.get();
    descr->activeSpmvBuffer = workspace.get();
    ASSERT_EQ(fixture.Execute(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(descr->activeBuffer, nullptr);
    EXPECT_EQ(descr->activeSpmvBuffer, nullptr);
    CheckWarmPreparedState(fixture, epoch);
    fixture.CheckValues();
    // A host identity call submits no kernel and leaves the other owner valid.
    fixture.alpha = 0.0F;
    fixture.beta = 1.0F;
    descr->activeBuffer = workspace.get();
    descr->activeSpmvBuffer = workspace.get();
    ASSERT_EQ(fixture.Execute(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(descr->activeBuffer, workspace.get());
    EXPECT_EQ(descr->activeSpmvBuffer, workspace.get());
    CheckWarmPreparedState(fixture, epoch);
    fixture.CheckValues();
}

#ifdef SDDMM_TEST_WITH_SPMM
void CheckWarmWorkspaceSpmm(aclrtStream stream, bool cube, bool alias)
{
    WarmWorkspaceFixture fixture(stream, cube);
    SpmmWorkspaceFixture spmm(fixture);
    ASSERT_GT(fixture.workspaceBytes, static_cast<size_t>(kSddmmWorkspaceBytes));
    DeviceBuffer workspace = DeviceBuffer::alloc(std::max(fixture.workspaceBytes, spmm.workspaceBytes));
    DeviceBuffer distinct = DeviceBuffer::alloc(alias ? 0 : spmm.workspaceBytes);
    void *spmmWorkspace = alias ? workspace.get() : distinct.get();
    ASSERT_EQ(fixture.Prepare(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    auto *descr = fixture.matC.get();
    const uint64_t epoch = descr->sddmmPrepared.preprocessEpoch;
    ASSERT_EQ(spmm.Prepare(spmmWorkspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(spmm.Execute(spmmWorkspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(descr->activeBuffer, spmmWorkspace);
    spmm.CheckValues(false);
    CheckWarmPreparedState(fixture, epoch);
    ASSERT_EQ(fixture.Execute(workspace.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    CheckWarmPreparedState(fixture, epoch);
    fixture.CheckValues();
    // Stop before launching SpMM with corrupt tiling if this regression recurs.
    ASSERT_EQ(descr->activeBuffer, alias ? nullptr : spmmWorkspace);
    ASSERT_EQ(spmm.Execute(spmmWorkspace), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    spmm.CheckValues(true);
}
#endif

} // namespace sddmm_extended
} // namespace sparse_test
