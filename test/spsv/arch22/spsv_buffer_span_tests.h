/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPSV_ARCH22_BUFFER_SPAN_TESTS_H_
#define SPSV_ARCH22_BUFFER_SPAN_TESTS_H_

// Included after Plan and shared helpers inside the test anonymous namespace.
void* AllocationTail(void* base, size_t bytes, size_t available)
{
    if (available == 0 || available > bytes)
        throw std::runtime_error("Invalid test allocation span");
    return static_cast<uint8_t*>(base) + bytes - available;
}

void RebindMatrixSpan(Plan& plan, int slot, void* pointer)
{
    void* arrays[] = {plan.dPtr.get(), plan.dIdx.get(), plan.dValues.get()};
    arrays[slot] = pointer;
    CheckSparse(aclsparseDestroySpMat(plan.mat), "destroy span matrix");
    plan.mat = nullptr;
    const auto& a = plan.a;
    const auto& c = plan.c;
    const auto base = ACL_SPARSE_INDEX_BASE_ZERO;
    aclsparseStatus_t status;
    if (c.format == ACL_SPARSE_FORMAT_CSR)
        status = aclsparseCreateCsr(&plan.mat, a.m, a.m, a.entries.size(), arrays[0], arrays[1], arrays[2],
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, c.type);
    else if (c.format == ACL_SPARSE_FORMAT_CSC)
        status = aclsparseCreateCsc(&plan.mat, a.m, a.m, a.entries.size(), arrays[0], arrays[1], arrays[2],
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, c.type);
    else if (c.format == ACL_SPARSE_FORMAT_COO)
        status = aclsparseCreateCoo(&plan.mat, a.m, a.m, a.entries.size(), arrays[0], arrays[1], arrays[2],
                                    ACL_SPARSE_INDEX_32I, base, c.type);
    else
        status = aclsparseCreateSlicedEll(&plan.mat, a.m, a.m, a.entries.size(), a.width, 1, arrays[0], arrays[1],
                                          arrays[2], ACL_SPARSE_INDEX_32I, base, c.type);
    CheckSparse(status, "create span matrix");
}

void CheckMatrixSpans(const Case& c, void* base, size_t bytes)
{
    Plan plan(MakeMatrix(c), c);
    const size_t sizes[] = {plan.a.ptr.size() * sizeof(int32_t), plan.a.idx.size() * sizeof(int32_t),
                            plan.initial.size() * sizeof(float)};
    const void* sources[] = {plan.a.ptr.data(), plan.a.idx.data(), plan.initial.data()};
    for (int slot = 0; slot < 3; ++slot) {
        SCOPED_TRACE("matrix_slot=" + std::to_string(slot));
        void* shortSpan = AllocationTail(base, bytes, sizes[slot] - sizeof(float));
        ASSERT_EQ(spsv_memory::ValidateRange(shortSpan, sizeof(float)), ACL_SUCCESS);
        ASSERT_EQ(spsv_memory::ValidateRange(shortSpan, sizes[slot]), ACL_ERROR_RT_PARAM_INVALID);
        RebindMatrixSpan(plan, slot, shortSpan);
        EXPECT_EQ(plan.Analyze(), kInvalid);
        void* exactSpan = AllocationTail(base, bytes, sizes[slot]);
        CheckAcl(aclrtMemcpy(exactSpan, sizes[slot], sources[slot], sizes[slot], ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy exact matrix span");
        RebindMatrixSpan(plan, slot, exactSpan);
        ASSERT_EQ(plan.Analyze(), kSuccess);
    }
}

void CheckVectorSpans(Plan& plan, void* base, size_t bytes)
{
    const size_t required = plan.input.size() * sizeof(float);
    for (int slot = 0; slot < 2; ++slot) {
        SCOPED_TRACE("vector_slot=" + std::to_string(slot));
        const auto vector = slot == 0 ? plan.x : plan.y;
        void* shortSpan = AllocationTail(base, bytes, required - sizeof(float));
        ASSERT_EQ(spsv_memory::ValidateRange(shortSpan, required), ACL_ERROR_RT_PARAM_INVALID);
        ASSERT_EQ(aclsparseDnVecSetValues(vector, shortSpan), kSuccess);
        EXPECT_EQ(plan.Solve(plan.c.operation), kInvalid);
        void* exactSpan = AllocationTail(base, bytes, required);
        CheckAcl(aclrtMemcpy(exactSpan, required, plan.input.data(), required, ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy exact vector span");
        ASSERT_EQ(aclsparseDnVecSetValues(vector, exactSpan), kSuccess);
        const auto solved = plan.Solve(plan.c.operation);
        const auto synced = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(solved, kSuccess);
        ASSERT_EQ(synced, ACL_SUCCESS);
        const void* output = slot == 0 ? plan.dY.get() : exactSpan;
        std::vector<float> actual(plan.input.size());
        CheckAcl(aclrtMemcpy(actual.data(), required, output, required, ACL_MEMCPY_DEVICE_TO_HOST),
                 "read exact vector output");
        Verify(actual,
               Golden(plan.a, plan.c, StoredValues(plan.a, plan.c, false), Input(plan.a, plan.c), plan.AlphaValue()),
               plan.c.type);
        ASSERT_EQ(aclsparseDnVecSetValues(vector, slot == 0 ? plan.dX.get() : plan.dY.get()), kSuccess);
    }
}

aclsparseStatus_t CallWithAlpha(Plan& plan, const void* alpha, int stage)
{
    if (stage == 0) {
        size_t bytes = 0;
        return aclsparseSpSV_bufferSize(SpSVArch22Test::handle, plan.c.operation, alpha, plan.mat, nullptr, nullptr,
                                        plan.c.type, kDefault, plan.state, &bytes);
    }
    if (stage == 1)
        return aclsparseSpSV_analysis(SpSVArch22Test::handle, plan.c.operation, alpha, plan.mat, nullptr, nullptr,
                                      plan.c.type, kDefault, plan.state, plan.workspace->get());
    return aclsparseSpSV_solve(SpSVArch22Test::handle, plan.c.operation, alpha, plan.mat, plan.x, plan.y, plan.c.type,
                               kDefault, plan.state);
}

void CheckDenseAndAlphaSpans(const Case& c, void* base, size_t bytes)
{
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    CheckVectorSpans(plan, base, bytes);
    if (::testing::Test::HasFatalFailure())
        return;
    const size_t required = plan.alpha.size() * sizeof(float);
    for (int stage = 0; stage < 3; ++stage) {
        SCOPED_TRACE("alpha_stage=" + std::to_string(stage));
        if (required > sizeof(float)) {
            void* shortSpan = AllocationTail(base, bytes, sizeof(float));
            ASSERT_EQ(spsv_memory::ValidateRange(shortSpan, required), ACL_ERROR_RT_PARAM_INVALID);
            EXPECT_EQ(CallWithAlpha(plan, shortSpan, stage), kInvalid);
        }
        void* exactSpan = AllocationTail(base, bytes, required);
        CheckAcl(aclrtMemcpy(exactSpan, required, plan.alpha.data(), required, ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy exact alpha span");
        const auto result = CallWithAlpha(plan, exactSpan, stage);
        const auto synced = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(result, kSuccess);
        ASSERT_EQ(synced, ACL_SUCCESS);
    }
}

void CheckUpdateSpans(const Case& c, void* base, size_t bytes)
{
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    const std::vector<Complex> diagonal(plan.a.m, Value(5.0, 0.25, c));
    for (const auto part : {ACL_SPARSE_SPSV_UPDATE_GENERAL, ACL_SPARSE_SPSV_UPDATE_DIAGONAL}) {
        SCOPED_TRACE("update_part=" + std::to_string(part));
        const auto packed = part == ACL_SPARSE_SPSV_UPDATE_GENERAL ? plan.initial : Pack(diagonal, c.type);
        const size_t required = packed.size() * sizeof(float);
        void* shortSpan = AllocationTail(base, bytes, required - sizeof(float));
        ASSERT_EQ(spsv_memory::ValidateRange(shortSpan, required), ACL_ERROR_RT_PARAM_INVALID);
        EXPECT_EQ(aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, shortSpan, part), kInvalid);
        void* exactSpan = AllocationTail(base, bytes, required);
        CheckAcl(aclrtMemcpy(exactSpan, required, packed.data(), required, ACL_MEMCPY_HOST_TO_DEVICE),
                 "copy exact update span");
        const auto result = aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, exactSpan, part);
        const auto synced = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(result, kSuccess);
        ASSERT_EQ(synced, ACL_SUCCESS);
        plan.VerifyTwice(Golden(plan.a, c, StoredValues(plan.a, c, false), Input(plan.a, c), plan.AlphaValue(),
                                part == ACL_SPARSE_SPSV_UPDATE_DIAGONAL ? &diagonal : nullptr));
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

void RunAllocationSpanCases(void (*check)(const Case&, void*, size_t))
{
    DeviceBuffer allocation(4096);
    void* base = nullptr;
    size_t bytes = 0;
    const auto result = aclrtMemGetAddressRange(allocation.get(), &base, &bytes);
    if (result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT)
        GTEST_SKIP() << "Public allocation range query unavailable";
    ASSERT_EQ(result, ACL_SUCCESS);
    ASSERT_EQ(spsv_memory::ValidateRangeResult(base, bytes, ACL_SUCCESS, base, bytes), ACL_SUCCESS);
    ASSERT_GE(bytes, 4096u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(base) % alignof(float), 0u);
    ASSERT_EQ(bytes % sizeof(float), 0u);
    constexpr aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC, ACL_SPARSE_FORMAT_COO,
                                             ACL_SPARSE_FORMAT_SLICED_ELL};
    for (const auto format : formats) {
        for (const auto type : {ACL_FLOAT, ACL_COMPLEX64}) {
            SCOPED_TRACE("format=" + std::to_string(format) + " type=" + std::to_string(type));
            Case c;
            c.format = format;
            c.type = type;
            c.deviceAlpha = true;
            check(c, base, bytes);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

TEST_F(SpSVArch22Test, AnalysisChecksCompleteMatrixSpans) { RunAllocationSpanCases(CheckMatrixSpans); }
TEST_F(SpSVArch22Test, SolveChecksCompleteVectorAndAlphaSpans) { RunAllocationSpanCases(CheckDenseAndAlphaSpans); }
TEST_F(SpSVArch22Test, UpdatesCheckStoredAndDiagonalValueSpans) { RunAllocationSpanCases(CheckUpdateSpans); }

#endif // SPSV_ARCH22_BUFFER_SPAN_TESTS_H_
