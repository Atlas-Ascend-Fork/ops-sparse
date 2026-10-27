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

#include <array>
#include <limits>

#include "sddmm_test_support.h"

namespace sparse_test {
namespace sddmm_extended {
namespace {

std::vector<aclsparseComplex> MakeComplexInfinityValues(size_t count)
{
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<aclsparseComplex, 8> values = {{
        {infinity, 1.0F}, {2.0F, infinity},
        {-infinity, -3.0F}, {4.0F, -infinity},
        {infinity, infinity}, {-infinity, infinity},
        {infinity, -infinity}, {-infinity, -infinity}}};
    std::vector<aclsparseComplex> initial(count);
    for (size_t i = 0; i < count; ++i) {
        initial[i] = values[i % values.size()];
    }
    return initial;
}

void InitializeCsrComplexInfinity(BsrTestResources &resources, int64_t reduction)
{
    constexpr int64_t rows = 2;
    constexpr int64_t columns = 64;
    const std::vector<int32_t> offsets = {0, 64, 128};
    std::vector<int32_t> indices(rows * columns);
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i % columns);
    }
    // A zero dense product leaves beta*C as the exact expected output while
    // alpha=one still executes the selected compute kernel and its epilogue.
    const std::vector<aclsparseComplex> x(rows * reduction, {0.0F, 0.0F});
    const std::vector<aclsparseComplex> y(reduction * columns, {0.0F, 0.0F});
    const auto initial = MakeComplexInfinityValues(indices.size());
    resources.dOffsets = SddmmGuardedBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
    resources.dColumns = SddmmGuardedBuffer::copyFrom(indices.data(), indices.size() * sizeof(int32_t));
    resources.dX = SddmmGuardedBuffer::copyFrom(x.data(), x.size() * sizeof(aclsparseComplex));
    resources.dY = SddmmGuardedBuffer::copyFrom(y.data(), y.size() * sizeof(aclsparseComplex));
    resources.dValues = SddmmGuardedBuffer::copyFrom(initial.data(), initial.size() * sizeof(aclsparseComplex));
    resources.matX = DnMatManager::create(rows, reduction, reduction,
        resources.dX.get(), ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW);
    resources.matY = DnMatManager::create(reduction, columns, columns,
        resources.dY.get(), ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW);
    CheckSetupStatus(aclsparseCreateCsr(resources.matC.put(), rows, columns,
        static_cast<int64_t>(indices.size()), resources.dOffsets.get(), resources.dColumns.get(),
        resources.dValues.get(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_COMPLEX64), "aclsparseCreateCsr");
}

} // namespace

void CheckCsrComplexBetaOneInfinity(aclrtStream stream, bool deviceScalars, int64_t reduction)
{
    BsrTestResources resources;
    InitializeCsrComplexInfinity(resources, reduction);
    const aclsparseComplex alpha{1.0F, 0.0F};
    const aclsparseComplex beta{1.0F, 0.0F};
    resources.SetScalars(stream, deviceScalars, alpha, beta);
    const auto stages = RunThreeStages(resources.handle, resources.alpha, resources.beta,
        resources.matX, resources.matY, resources.matC.get(), ACL_COMPLEX64, stream, resources.workspace);
    ASSERT_TRUE(StagesSucceeded(stages));
    if (reduction == 128 && !deviceScalars) {
        const auto flags = resources.matC.get()->sddmmPrepared.patternFlags;
        ASSERT_NE(flags & SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
    }
    // Compare bytes: Inf in either component must not contaminate the other.
    EXPECT_TRUE(resources.dValues.unchanged());
    EXPECT_TRUE(resources.dValues.guardsIntact());
}

void CheckBsrComplexBetaOneInfinity(aclrtStream stream, int64_t blockSize,
    int64_t reduction, aclsparseOrder_t order, bool deviceScalars)
{
    BsrComplexCase testCase;
    testCase.blockRows = 2;
    testCase.blockCols = 4;
    testCase.blockSize = blockSize;
    testCase.k = reduction;
    testCase.zeroBasedOffsets = {0, 2, 4};
    testCase.zeroBasedColumns = {0, 1, 2, 3};
    testCase.blockOrder = order;
    testCase.deviceScalars = deviceScalars;
    testCase.alpha = {1.0F, 0.0F};
    testCase.beta = {1.0F, 0.0F};
    const std::vector<aclsparseComplex> x(testCase.blockRows * blockSize * reduction, {0.0F, 0.0F});
    const std::vector<aclsparseComplex> y(reduction * testCase.blockCols * blockSize, {0.0F, 0.0F});
    const auto initial = MakeComplexInfinityValues(testCase.zeroBasedColumns.size() * blockSize * blockSize);
    BsrWorkspaceFixture fixture(stream);
    fixture.Initialize(testCase, ACL_COMPLEX64, x, y, initial);
    ASSERT_TRUE(fixture.Query());
    if (blockSize == 16 && !deviceScalars) {
        ASSERT_GT(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
        constexpr auto alignment = sddmm_arch35::kSddmmComplexBsrCubeInputAlignment;
        ASSERT_EQ(reinterpret_cast<uintptr_t>(fixture.dX.get()) % alignment, 0U);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(fixture.dY.get()) % alignment, 0U);
    }
    fixture.guardedWorkspace = SddmmGuardedBuffer::copyFrom(nullptr, fixture.stages.workspaceSize);
    ASSERT_TRUE(fixture.Run());
    EXPECT_TRUE(fixture.dValues.unchanged());
}

} // namespace sddmm_extended
} // namespace sparse_test
