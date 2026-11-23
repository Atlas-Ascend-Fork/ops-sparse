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

#ifndef SPSV_DESCRIPTOR_REUSE_TESTS_H_
#define SPSV_DESCRIPTOR_REUSE_TESTS_H_

TEST_F(SpSVExceptionTest, ReviewInPlaceDescriptorReuse) {
    // Cover chain, grouped and cross-block schedules. The widest case stays
    // above the single-block threshold and rotates predecessor ownership.
    for (const int32_t width : {1, 32, 256, 1024}) {
        const int32_t m = 5 * width;
        CsrMatrix a;
        a.rows = a.cols = m;
        a.rowOffsets.push_back(0);
        std::vector<float> x(m), expected(m), currentX(m);
        BuildDescriptorReuseMatrix(a, x, expected, width, m);
        a.nnz = static_cast<int64_t>(a.values.size());
        for (bool inplace : {false, true}) {
            SpSVNpuConfig cfg;
            auto setup = SetupMatrixDescriptor(a, cfg);
            auto dX = DeviceBuffer::copyFrom(x.data(), m * sizeof(float));
            auto dY = DeviceBuffer::alloc(m * sizeof(float));
            auto vecX = DnVecManager::createConst(m, dX.get(), ACL_FLOAT);
            auto vecY = DnVecManager::create(m, inplace ? dX.get() : dY.get(), ACL_FLOAT);
            SpSVDescrManager descr;
            float alpha = 1.0f;
            auto workspace = RunAnalysisT(spHandle_->get(), cfg, alpha, setup.matA,
                                          vecX, vecY, descr, ACL_FLOAT);
            // Keep analysis/descriptors alive, but change the RHS every call.
            // This also detects cached input or solution words from a prior solve.
            for (int repeat = 0; repeat < 100; ++repeat) {
                const float factor = (repeat % 2 == 0) ? 1.0f : -2.0f;
                std::transform(x.begin(), x.end(), currentX.begin(),
                    [factor](float value) { return factor * value; });
                ASSERT_EQ(aclrtMemcpy(dX.get(), m * sizeof(float), currentX.data(),
                                      m * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
                std::vector<float> y(m);
                RunSolveT(spHandle_->get(), env_->stream(), cfg, alpha, setup.matA,
                          vecX, vecY, descr, inplace ? dX : dY, y, ACL_FLOAT);
                for (int32_t row = 0; row < m; ++row) {
                    int32_t previous = row >= width
                        ? (row / width - 1) * width + (row + 127) % width : -1;
                    ASSERT_EQ(y[row], factor * expected[row])
                        << width << ":" << inplace << ":" << repeat << ":" << row
                        << " x=" << currentX[row] << " prev=" << previous
                        << " prevExpected=" << (previous >= 0 ? expected[previous] : 0.0f)
                        << " prevActual=" << (previous >= 0 ? y[previous] : 0.0f);
                }
            }
        }
    }
}

TEST_F(SpSVExceptionTest, L2_destroy_null) {
    // null 幂等契约：spsv_host.cpp destroyDescr 对 nullptr 无操作返回 SUCCESS，
    // 算子 README 约束说明同款口径；此前断言为恒真式（== SUCCESS || != SUCCESS
    // 恒成立），对任意返回值均通过，未起到回归防护作用，收紧为精确断言。
    auto ret = aclsparseSpSV_destroyDescr(nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_solve_before_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, L2_invalid_opA) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), static_cast<aclsparseOperation_t>(999), &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_null_vecX_solve) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        nullptr, s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_null_vecY_solve) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    ret = aclsparseSpSV_solve(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), nullptr, ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get());
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_null_externalBuffer_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = aclsparseSpSV_analysis(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SpSVExceptionTest, WB_non_square_matrix) {
    int64_t rows = 4, cols = 8;
    auto csr = makeSparseCsr(rows, cols, 0.5, 42);
    auto dRowOff = DeviceBuffer::copyFrom(csr.rowOffsets.data(),
                                           csr.rowOffsets.size() * sizeof(int32_t));
    auto dColIdx = DeviceBuffer::copyFrom(csr.colIndices.data(),
                                           csr.colIndices.size() * sizeof(int32_t));
    auto dVals = DeviceBuffer::copyFrom(csr.values.data(),
                                         csr.values.size() * sizeof(float));
    auto matA = SpMatManager::createCsr(rows, cols, csr.nnz,
        dRowOff.get(), dColIdx.get(), dVals.get());
    std::vector<float> x(rows, 1.0f), y(rows, 0.0f);
    auto dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
    auto dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));
    auto vecX = DnVecManager::createConst(rows, dX.get(), ACL_FLOAT);
    auto vecY = DnVecManager::create(rows, dY.get(), ACL_FLOAT);
    SpSVDescrManager spsvDescr;
    float alpha = 1.0f;
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, matA.cget(),
        vecX.cget(), vecY.get(), ACL_FLOAT,
        ACL_SPARSE_SPSV_ALG_DEFAULT, spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_unsupported_computeType) {
    auto s = BasicSetup::Make(spHandle_->get());
    size_t bufSize = 0;
    auto ret = aclsparseSpSV_bufferSize(
        spHandle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, &s.alpha, s.matA.cget(),
        s.vecX.cget(), s.vecY.get(), ACL_FLOAT16,
        ACL_SPARSE_SPSV_ALG_DEFAULT, s.spsvDescr.get(), &bufSize);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_update_before_analysis) {
    auto s = BasicSetup::Make(spHandle_->get());
    std::vector<float> newVals(4, 1.0f);
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    auto ret = aclsparseSpSV_updateMatrix(
        spHandle_->get(), s.spsvDescr.get(), dNewVals.get(),
        ACL_SPARSE_SPSV_UPDATE_GENERAL);
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

TEST_F(SpSVExceptionTest, WB_unsupported_updatePart) {
    auto s = BasicSetup::Make(spHandle_->get());
    auto ret = RunSpSVPipeline(spHandle_->get(), s);
    ASSERT_EQ(ret, ACL_SPARSE_STATUS_SUCCESS);
    std::vector<float> newVals(4, 1.0f);
    auto dNewVals = DeviceBuffer::copyFrom(newVals.data(), newVals.size() * sizeof(float));
    ret = aclsparseSpSV_updateMatrix(
        spHandle_->get(), s.spsvDescr.get(), dNewVals.get(),
        static_cast<aclsparseSpSVUpdate_t>(999));
    EXPECT_NE(ret, ACL_SPARSE_STATUS_SUCCESS);
}

}


#endif // SPSV_DESCRIPTOR_REUSE_TESTS_H_
