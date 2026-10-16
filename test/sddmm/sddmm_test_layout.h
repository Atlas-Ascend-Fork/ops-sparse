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
#ifndef TEST_SDDMM_SDDMM_TEST_LAYOUT_H_
#define TEST_SDDMM_SDDMM_TEST_LAYOUT_H_

#include <random>
#include <vector>

#include "test_common.h"
#include "sddmm_golden.h"
#include "sddmm_npu_wrapper.h"
#include "sddmm_param.h"

namespace sparse_test {

inline std::vector<double> Fp16ToDoubles(const std::vector<uint16_t>& values) {
    std::vector<double> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<double>(Fp16BitsToFp32(values[i]));
    }
    return result;
}

inline std::vector<double> QuantizeDoublesViaFp16(const std::vector<double>& values) {
    return Fp16ToDoubles(DoublesToFp16(values));
}

template <typename T>
inline std::vector<T> RepackRowToCol(const std::vector<T>& rowBuffer,
                                     int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return rowBuffer;
    }
    std::vector<T> colBuffer(rowBuffer.size());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            colBuffer[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] =
                rowBuffer[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
        }
    }
    return colBuffer;
}

class SddmmFunctionalTestFixture : public testing::TestWithParam<SddmmTestParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SddmmTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

struct SddmmInputs {
    SddmmCsr csrC;
    std::vector<double> Xf64;
    std::vector<double> Yf64;
    int64_t nnz = 0;
};

inline SddmmInputs GenerateSddmmInputs(const SddmmTestParam& param) {
    SddmmInputs inputs;
    inputs.csrC = MakeSddmmSparsity(param.m, param.n, param.sparsity_ratio,
                                    param.value_lo, param.value_hi, param.random_seed);
    inputs.nnz = inputs.csrC.nnz;
    std::mt19937 rngX(param.random_seed + 1);
    std::mt19937 rngY(param.random_seed + 2);
    std::uniform_real_distribution<double> distribution(param.value_lo, param.value_hi);
    inputs.Xf64.resize(static_cast<size_t>(param.m) * static_cast<size_t>(param.k));
    inputs.Yf64.resize(static_cast<size_t>(param.n) * static_cast<size_t>(param.k));
    for (double& value : inputs.Xf64) value = distribution(rngX);
    for (double& value : inputs.Yf64) value = distribution(rngY);
    return inputs;
}

inline std::vector<double> ComputeGoldenExpect(const SddmmTestParam& param,
                                                const SddmmCsr& csrC,
                                                const std::vector<double>& x,
                                                const std::vector<double>& y,
                                                aclDataType dtype,
                                                aclsparseOperation_t opX,
                                                aclsparseOperation_t opY) {
    std::vector<double> goldenX = x;
    std::vector<double> goldenY = y;
    SddmmCsr goldenCsr = csrC;
    double alpha = param.alpha;
    double beta = param.beta;
    if (dtype == ACL_FLOAT16) {
        goldenX = QuantizeDoublesViaFp16(x);
        goldenY = QuantizeDoublesViaFp16(y);
        goldenCsr.values = QuantizeDoublesViaFp16(csrC.values);
        alpha = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(param.alpha))));
        beta = static_cast<double>(Fp16BitsToFp32(Fp32ToFp16Bits(static_cast<float>(param.beta))));
    }
    return SddmmGolden(param.m, param.n, param.k, goldenX, goldenY, goldenCsr, alpha, beta, opX, opY);
}

template <typename T>
inline SddmmNpuResult RunNpuSddmmTyped(HandleManager& handle, aclrtStream stream,
                                       const SddmmTestParam& param, const SddmmCsr& csrC,
                                       std::vector<T> cValues, std::vector<T> xValues,
                                       std::vector<T> yValues, float alpha, float beta,
                                       aclDataType dtype, aclDataType computeType,
                                       aclsparseSDDMMAlg_t alg, aclsparseOperation_t opX,
                                       aclsparseOperation_t opY, aclsparseOrder_t orderX,
                                       aclsparseOrder_t orderY, int64_t nnz) {
    const int64_t xRows = opX == ACL_SPARSE_OP_TRANSPOSE ? param.k : param.m;
    const int64_t xCols = opX == ACL_SPARSE_OP_TRANSPOSE ? param.m : param.k;
    const int64_t yRows = opY == ACL_SPARSE_OP_TRANSPOSE ? param.n : param.k;
    const int64_t yCols = opY == ACL_SPARSE_OP_TRANSPOSE ? param.k : param.n;
    if (orderX == ACL_SPARSE_ORDER_COL) {
        xValues = RepackRowToCol(xValues, xRows, xCols);
    }
    if (orderY == ACL_SPARSE_ORDER_COL) {
        yValues = RepackRowToCol(yValues, yRows, yCols);
    }
    return SddmmNpu<T>(handle, stream, param.m, param.n, param.k, opX, opY, alpha, beta,
                       dtype, computeType, alg, orderX, orderY, csrC.rowOffsets,
                       csrC.colIndices, cValues, xValues, yValues, nnz);
}

inline SddmmNpuResult RunNpuSddmm(const SddmmTestParam& param, aclrtStream stream,
                                   const SddmmCsr& csrC, const std::vector<double>& x,
                                   const std::vector<double>& y, aclDataType dtype,
                                   aclDataType computeType, aclsparseSDDMMAlg_t alg,
                                   aclsparseOperation_t opX, aclsparseOperation_t opY,
                                   aclsparseOrder_t orderX, aclsparseOrder_t orderY,
                                   int64_t nnz) {
    HandleManager handle;
    const float alpha = static_cast<float>(param.alpha);
    const float beta = static_cast<float>(param.beta);
    if (dtype == ACL_FLOAT) {
        return RunNpuSddmmTyped<float>(handle, stream, param, csrC, DoublesToFp32(csrC.values),
                                       DoublesToFp32(x), DoublesToFp32(y), alpha, beta, dtype,
                                       computeType, alg, opX, opY, orderX, orderY, nnz);
    }
    return RunNpuSddmmTyped<uint16_t>(handle, stream, param, csrC, DoublesToFp16(csrC.values),
                                      DoublesToFp16(x), DoublesToFp16(y), alpha, beta, dtype,
                                      computeType, alg, opX, opY, orderX, orderY, nnz);
}

inline bool VerifySddmmPrecisionCore(const std::string& caseName,
                                     const SddmmNpuResult& npuResult,
                                     const std::vector<double>& expectFp64,
                                     aclDataType dtype, int64_t nnz) {
    std::vector<float> golden(static_cast<size_t>(nnz));
    std::vector<float> actual(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        float value = static_cast<float>(expectFp64[static_cast<size_t>(i)]);
        if (dtype == ACL_FLOAT16) {
            value = value > 65504.0f ? 65504.0f : (value < -65504.0f ? -65504.0f : value);
        }
        golden[static_cast<size_t>(i)] = value;
        actual[static_cast<size_t>(i)] = static_cast<float>(npuResult.valuesOut[static_cast<size_t>(i)]);
    }
    VerifyConfig config;
    applyMixedTolerance(config, dtype, golden.data(), golden.size());
    return Verifier::verifyVector(actual.data(), golden.data(), static_cast<size_t>(nnz), 1, config, caseName);
}


}  // namespace sparse_test

#endif  // TEST_SDDMM_SDDMM_TEST_LAYOUT_H_
