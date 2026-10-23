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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <acl/acl.h>
#include <cann_ops_sparse.h>
#include "spmm_test_utils.h"

namespace {

using spmm_test::CheckAcl;
using spmm_test::CheckSparse;
using spmm_test::Bf16ToFloat;
using spmm_test::FloatToBf16;
using spmm_test::FloatToHalf;
using spmm_test::HalfToFloat;

bool CopyToDevice(void *dst, const void *src, size_t bytes)
{
    return CheckAcl(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
}

bool CopyToHost(void *dst, const void *src, size_t bytes)
{
    return CheckAcl(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
}

struct RealCaseData {
    std::vector<int32_t> rowOffsets{0, 2, 3, 5};
    std::vector<int32_t> colIndices{0, 2, 1, 0, 3};
    std::vector<float> aSource{1.25f, -0.5f, 0.75f, -1.0f, 0.375f};
    std::vector<float> bSource;
    std::vector<float> cSource;
    std::vector<float> expected;
    std::vector<uint16_t> a16;
    std::vector<uint16_t> b16;
    std::vector<uint16_t> c16;
};

struct RealDevice {
    void *rows = nullptr;
    void *cols = nullptr;
    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
    void *workspace = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t matB = nullptr;
    aclsparseDnMatDescr_t matC = nullptr;
};

void CleanupRealDevice(RealDevice *device)
{
    if (device->matB) aclsparseDestroyDnMat(device->matB);
    if (device->matC) aclsparseDestroyDnMat(device->matC);
    if (device->matA) aclsparseDestroySpMat(device->matA);
    if (device->handle) aclsparseDestroy(device->handle);
    if (device->workspace) aclrtFree(device->workspace);
    if (device->rows) aclrtFree(device->rows);
    if (device->cols) aclrtFree(device->cols);
    if (device->a) aclrtFree(device->a);
    if (device->b) aclrtFree(device->b);
    if (device->c) aclrtFree(device->c);
}

void InitializeRealSources(RealCaseData *data)
{
    constexpr int64_t k = 4;
    constexpr int64_t n = 5;
    data->bSource.resize(static_cast<size_t>(k * n));
    data->cSource.resize(static_cast<size_t>(3 * n));
    for (size_t i = 0; i < data->bSource.size(); ++i) {
        data->bSource[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.25f;
    }
    for (size_t i = 0; i < data->cSource.size(); ++i) {
        data->cSource[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    }
}

float QuantizeReal(float value, aclDataType type)
{
    if (type == ACL_FLOAT) return value;
    return type == ACL_BF16 ? Bf16ToFloat(FloatToBf16(value)) :
        HalfToFloat(FloatToHalf(value));
}

void QuantizeRealInputs(RealCaseData *data, aclDataType inputType,
                        aclDataType outputType)
{
    const bool bf16Input = inputType == ACL_BF16;
    data->a16.resize(data->aSource.size());
    data->b16.resize(data->bSource.size());
    for (size_t i = 0; i < data->a16.size(); ++i) {
        data->a16[i] = bf16Input ? FloatToBf16(data->aSource[i]) : FloatToHalf(data->aSource[i]);
        data->aSource[i] = QuantizeReal(data->aSource[i], inputType);
    }
    for (size_t i = 0; i < data->b16.size(); ++i) {
        data->b16[i] = bf16Input ? FloatToBf16(data->bSource[i]) : FloatToHalf(data->bSource[i]);
        data->bSource[i] = QuantizeReal(data->bSource[i], inputType);
    }
    if (outputType == ACL_FLOAT) return;
    data->c16.resize(data->cSource.size());
    for (size_t i = 0; i < data->c16.size(); ++i) {
        data->c16[i] = outputType == ACL_BF16 ? FloatToBf16(data->cSource[i]) : FloatToHalf(data->cSource[i]);
        data->cSource[i] = QuantizeReal(data->cSource[i], outputType);
    }
}

void BuildRealExpected(RealCaseData *data, aclDataType outputType)
{
    constexpr int64_t m = 3;
    constexpr int64_t n = 5;
    constexpr float alpha = 0.75f;
    constexpr float beta = -0.25f;
    data->expected = data->cSource;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int32_t p = data->rowOffsets[row]; p < data->rowOffsets[row + 1]; ++p) {
                sum += data->aSource[p] * data->bSource[
                    static_cast<size_t>(data->colIndices[p] * n + col)];
            }
            const size_t output = static_cast<size_t>(row * n + col);
            data->expected[output] = QuantizeReal(
                alpha * sum + beta * data->expected[output], outputType);
        }
    }
}

bool AllocateRealDevice(const RealCaseData &data, aclDataType outputType,
                        RealDevice *device)
{
    const size_t cBytes = data.cSource.size() *
        (outputType == ACL_FLOAT ? sizeof(float) : sizeof(uint16_t));
    return CheckAcl(aclrtMalloc(&device->rows, data.rowOffsets.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&device->cols, data.colIndices.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc cols") &&
        CheckAcl(aclrtMalloc(&device->a, data.a16.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc A") &&
        CheckAcl(aclrtMalloc(&device->b, data.b16.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&device->c, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
}

bool UploadRealInput(const RealCaseData &data, aclDataType outputType,
                     const RealDevice &device)
{
    const void *output = outputType == ACL_FLOAT ?
        static_cast<const void *>(data.cSource.data()) : static_cast<const void *>(data.c16.data());
    const size_t cBytes = data.cSource.size() *
        (outputType == ACL_FLOAT ? sizeof(float) : sizeof(uint16_t));
    return CopyToDevice(device.rows, data.rowOffsets.data(), data.rowOffsets.size() * sizeof(int32_t)) &&
        CopyToDevice(device.cols, data.colIndices.data(), data.colIndices.size() * sizeof(int32_t)) &&
        CopyToDevice(device.a, data.a16.data(), data.a16.size() * sizeof(uint16_t)) &&
        CopyToDevice(device.b, data.b16.data(), data.b16.size() * sizeof(uint16_t)) &&
        CopyToDevice(device.c, output, cBytes);
}

bool RunRealSpmm(aclrtStream stream, const RealCaseData &data,
                 aclDataType inputType, aclDataType outputType,
                 aclsparseSpMMAlg_t alg, RealDevice *device)
{
    constexpr int64_t m = 3;
    constexpr int64_t k = 4;
    constexpr int64_t n = 5;
    constexpr float alpha = 0.75f;
    constexpr float beta = -0.25f;
    size_t workspaceSize = 0;
    return CheckSparse(aclsparseCreate(&device->handle), "create handle") &&
        CheckSparse(aclsparseSetStream(device->handle, stream), "set stream") &&
        CheckSparse(aclsparseCreateCsr(&device->matA, m, k, data.colIndices.size(),
            device->rows, device->cols, device->a,
            ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, inputType), "create CSR") &&
        CheckSparse(aclsparseCreateDnMat(&device->matB, k, n, n, device->b, inputType, ACL_SPARSE_ORDER_ROW), "create B") &&
        CheckSparse(aclsparseCreateDnMat(&device->matC, m, n, n, device->c, outputType, ACL_SPARSE_ORDER_ROW), "create C") &&
        CheckSparse(aclsparseSpMMGetBufferSize(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, device->matA, device->matB, &beta, device->matC,
            ACL_FLOAT, alg, &workspaceSize), "buffer size") &&
        CheckAcl(aclrtMalloc(&device->workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace") &&
        CheckSparse(aclsparseSpMMPreprocess(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, device->matA, device->matB, &beta, device->matC,
            ACL_FLOAT, alg, device->workspace), "preprocess") &&
        CheckSparse(aclsparseSpMM(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, &alpha, device->matA, device->matB, &beta, device->matC,
            ACL_FLOAT, alg, device->workspace), "execute") &&
        CheckAcl(aclrtSynchronizeStream(stream), "synchronize");
}

bool VerifyRealResult(const RealCaseData &data, aclDataType outputType,
                      const RealDevice &device, const char *name)
{
    std::vector<float> actual(data.cSource.size());
    if (outputType == ACL_FLOAT) {
        if (!CopyToHost(actual.data(), device.c, actual.size() * sizeof(float))) return false;
    } else {
        std::vector<uint16_t> raw(actual.size());
        if (!CopyToHost(raw.data(), device.c, raw.size() * sizeof(uint16_t))) return false;
        for (size_t i = 0; i < raw.size(); ++i) {
            actual[i] = outputType == ACL_BF16 ? Bf16ToFloat(raw[i]) : HalfToFloat(raw[i]);
        }
    }
    const float tolerance = outputType == ACL_BF16 ? 0.015625f :
        (outputType == ACL_FLOAT16 ? 0.001953125f : 1.0e-6f);
    bool pass = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - data.expected[i]) > tolerance) {
            std::printf("%s mismatch[%zu]: expected=%g actual=%g\n", name, i, data.expected[i], actual[i]);
            pass = false;
        }
    }
    std::printf("%s: %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunRealCase(aclrtStream stream, const char *name, aclDataType inputType,
                 aclDataType outputType, aclsparseSpMMAlg_t alg)
{
    RealCaseData data;
    RealDevice device;
    InitializeRealSources(&data);
    QuantizeRealInputs(&data, inputType, outputType);
    BuildRealExpected(&data, outputType);
    const bool pass = AllocateRealDevice(data, outputType, &device) &&
        UploadRealInput(data, outputType, device) &&
        RunRealSpmm(stream, data, inputType, outputType, alg, &device) &&
        VerifyRealResult(data, outputType, device, name);
    CleanupRealDevice(&device);
    return pass;
}

std::complex<float> ToComplex(aclsparseComplex value)
{
    return {value.x, value.y};
}

aclsparseComplex FromComplex(std::complex<float> value)
{
    return {value.real(), value.imag()};
}

struct ComplexCaseData {
    std::vector<int32_t> rows{0, 2, 4};
    std::vector<int32_t> columns{0, 2, 1, 2};
    std::vector<aclsparseComplex> values{{1.0f, 0.5f}, {-0.25f, 1.0f},
                                         {0.75f, -0.5f}, {1.25f, 0.25f}};
    std::vector<aclsparseComplex> matB;
    std::vector<aclsparseComplex> matC{{0.5f, -0.25f}, {-0.75f, 0.5f},
                                        {0.25f, 0.75f}, {1.0f, -0.5f}};
    aclsparseComplex alpha{0.75f, 0.25f};
    aclsparseComplex beta{-0.5f, 0.125f};
    std::vector<std::complex<float>> expected;
};

struct ComplexDevice {
    void *rows = nullptr;
    void *columns = nullptr;
    void *values = nullptr;
    void *matB = nullptr;
    void *matC = nullptr;
    void *alpha = nullptr;
    void *beta = nullptr;
    void *workspace = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpMatDescr_t matA = nullptr;
    aclsparseDnMatDescr_t denseB = nullptr;
    aclsparseDnMatDescr_t denseC = nullptr;
};

void CleanupComplexDevice(ComplexDevice *device)
{
    if (device->denseB) aclsparseDestroyDnMat(device->denseB);
    if (device->denseC) aclsparseDestroyDnMat(device->denseC);
    if (device->matA) aclsparseDestroySpMat(device->matA);
    if (device->handle) aclsparseDestroy(device->handle);
    if (device->workspace) aclrtFree(device->workspace);
    if (device->rows) aclrtFree(device->rows);
    if (device->columns) aclrtFree(device->columns);
    if (device->values) aclrtFree(device->values);
    if (device->matB) aclrtFree(device->matB);
    if (device->matC) aclrtFree(device->matC);
    if (device->alpha) aclrtFree(device->alpha);
    if (device->beta) aclrtFree(device->beta);
}

void BuildComplexMatB(ComplexCaseData *data, aclsparseOperation_t opB)
{
    if (opB == ACL_SPARSE_OP_NON_TRANSPOSE) {
        data->matB = {{1.0f, 0.25f}, {-0.5f, 0.75f}, {0.25f, -1.0f},
                      {1.5f, 0.5f}, {-0.75f, 0.25f}, {0.5f, -0.5f}};
        return;
    }
    // Stored B is [N,K]; op(B)=conj(B^T) is [K,N].
    data->matB = {{1.0f, -0.25f}, {0.25f, 1.0f}, {-0.75f, -0.25f},
                  {-0.5f, -0.75f}, {1.5f, -0.5f}, {0.5f, 0.5f}};
}

void BuildComplexExpected(ComplexCaseData *data, aclsparseOperation_t opB)
{
    constexpr int64_t m = 2;
    constexpr int64_t k = 3;
    constexpr int64_t n = 2;
    data->expected.resize(data->matC.size());
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            std::complex<float> sum{};
            for (int32_t p = data->rows[row]; p < data->rows[row + 1]; ++p) {
                const int32_t inner = data->columns[p];
                std::complex<float> bv;
                if (opB == ACL_SPARSE_OP_NON_TRANSPOSE) {
                    bv = ToComplex(data->matB[static_cast<size_t>(inner * n + col)]);
                } else {
                    bv = std::conj(ToComplex(data->matB[static_cast<size_t>(col * k + inner)]));
                }
                sum += ToComplex(data->values[p]) * bv;
            }
            const size_t index = static_cast<size_t>(row * n + col);
            data->expected[index] = ToComplex(data->alpha) * sum +
                ToComplex(data->beta) * ToComplex(data->matC[index]);
        }
    }
}

bool AllocateComplexDevice(const ComplexCaseData &data, ComplexDevice *device)
{
    return CheckAcl(aclrtMalloc(&device->rows, data.rows.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc rows") &&
        CheckAcl(aclrtMalloc(&device->columns, data.columns.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "malloc cols") &&
        CheckAcl(aclrtMalloc(&device->values, data.values.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc A") &&
        CheckAcl(aclrtMalloc(&device->matB, data.matB.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc B") &&
        CheckAcl(aclrtMalloc(&device->matC, data.matC.size() * sizeof(aclsparseComplex), ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
}

bool UploadComplexInput(const ComplexCaseData &data, const ComplexDevice &device)
{
    return CopyToDevice(device.rows, data.rows.data(), data.rows.size() * sizeof(int32_t)) &&
        CopyToDevice(device.columns, data.columns.data(), data.columns.size() * sizeof(int32_t)) &&
        CopyToDevice(device.values, data.values.data(), data.values.size() * sizeof(aclsparseComplex)) &&
        CopyToDevice(device.matB, data.matB.data(), data.matB.size() * sizeof(aclsparseComplex)) &&
        CopyToDevice(device.matC, data.matC.data(), data.matC.size() * sizeof(aclsparseComplex));
}

bool PrepareComplexScalars(const ComplexCaseData &data, bool deviceScalars,
                           ComplexDevice *device)
{
    if (!deviceScalars) return true;
    return CheckAcl(aclrtMalloc(&device->alpha, sizeof(data.alpha), ACL_MEM_MALLOC_HUGE_FIRST), "malloc alpha") &&
        CheckAcl(aclrtMalloc(&device->beta, sizeof(data.beta), ACL_MEM_MALLOC_HUGE_FIRST), "malloc beta") &&
        CopyToDevice(device->alpha, &data.alpha, sizeof(data.alpha)) &&
        CopyToDevice(device->beta, &data.beta, sizeof(data.beta));
}

const void *ComplexAlpha(const ComplexCaseData &data, const ComplexDevice &device,
                         bool deviceScalars)
{
    return deviceScalars ? device.alpha : &data.alpha;
}

const void *ComplexBeta(const ComplexCaseData &data, const ComplexDevice &device,
                        bool deviceScalars)
{
    return deviceScalars ? device.beta : &data.beta;
}

enum class ComplexPrepareResult { kReady, kExpectedRejection, kFailure };

ComplexPrepareResult PrepareComplexSpmm(aclrtStream stream, const ComplexCaseData &data,
                                        aclsparseOperation_t opB, aclsparseSpMMAlg_t alg,
                                        bool deviceScalars, const char *name,
                                        ComplexDevice *device)
{
    constexpr int64_t m = 2;
    constexpr int64_t k = 3;
    constexpr int64_t n = 2;
    const int64_t bRows = opB == ACL_SPARSE_OP_NON_TRANSPOSE ? k : n;
    const int64_t bCols = opB == ACL_SPARSE_OP_NON_TRANSPOSE ? n : k;
    const void *alpha = ComplexAlpha(data, *device, deviceScalars);
    const void *beta = ComplexBeta(data, *device, deviceScalars);
    if (!CheckSparse(aclsparseCreate(&device->handle), "create handle") ||
        !CheckSparse(aclsparseSetStream(device->handle, stream), "set stream") ||
        (deviceScalars && !CheckSparse(aclsparseSetPointerMode(device->handle, ACL_SPARSE_POINTER_MODE_DEVICE), "pointer mode")) ||
        !CheckSparse(aclsparseCreateCsr(&device->matA, m, k, data.columns.size(), device->rows,
            device->columns, device->values, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_BASE_ZERO, ACL_COMPLEX64), "create CSR") ||
        !CheckSparse(aclsparseCreateDnMat(&device->denseB, bRows, bCols, bCols, device->matB,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create B") ||
        !CheckSparse(aclsparseCreateDnMat(&device->denseC, m, n, n, device->matC,
            ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW), "create C")) return ComplexPrepareResult::kFailure;

    size_t workspaceSize = 0;
    const aclsparseStatus_t sizeStatus = aclsparseSpMMGetBufferSize(device->handle,
        ACL_SPARSE_OP_NON_TRANSPOSE, opB, alpha, device->matA, device->denseB, beta,
        device->denseC, ACL_COMPLEX64, alg, &workspaceSize);
    if (alg == ACL_SPARSE_SPMM_CSR_ALG3 && opB == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) {
        const bool pass = sizeStatus == ACL_SPARSE_STATUS_NOT_SUPPORTED;
        std::printf("%s: %s\n", name, pass ? "PASS" : "FAIL");
        return pass ? ComplexPrepareResult::kExpectedRejection : ComplexPrepareResult::kFailure;
    }
    if (!CheckSparse(sizeStatus, "buffer size") ||
        !CheckAcl(aclrtMalloc(&device->workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace") ||
        !CheckSparse(aclsparseSpMMPreprocess(device->handle, ACL_SPARSE_OP_NON_TRANSPOSE, opB,
            alpha, device->matA, device->denseB, beta, device->denseC, ACL_COMPLEX64, alg,
            device->workspace), "preprocess")) return ComplexPrepareResult::kFailure;
    return ComplexPrepareResult::kReady;
}

bool ExecuteComplexSpmm(aclrtStream stream, ComplexCaseData *data,
                        aclsparseOperation_t opB, aclsparseSpMMAlg_t alg,
                        bool deviceScalars, const ComplexDevice &device)
{
    return CheckSparse(aclsparseSpMM(device.handle, ACL_SPARSE_OP_NON_TRANSPOSE, opB,
        ComplexAlpha(*data, device, deviceScalars), device.matA, device.denseB,
        ComplexBeta(*data, device, deviceScalars), device.denseC, ACL_COMPLEX64, alg,
        device.workspace), "execute") && CheckAcl(aclrtSynchronizeStream(stream), "synchronize") &&
        CopyToHost(data->matC.data(), device.matC,
                   data->matC.size() * sizeof(aclsparseComplex));
}

bool VerifyComplexResult(const ComplexCaseData &data, const char *name)
{
    bool pass = true;
    for (size_t i = 0; i < data.matC.size(); ++i) {
        const std::complex<float> actual = ToComplex(data.matC[i]);
        if (std::abs(actual - data.expected[i]) > 2.0e-5f) {
            std::printf("%s mismatch[%zu]: expected=(%g,%g) actual=(%g,%g)\n",
                        name, i, data.expected[i].real(), data.expected[i].imag(), actual.real(), actual.imag());
            pass = false;
        }
    }
    std::printf("%s: %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

bool RunComplexCase(aclrtStream stream, const char *name, aclsparseOperation_t opB,
                    aclsparseSpMMAlg_t alg, bool deviceScalars)
{
    ComplexCaseData data;
    ComplexDevice device;
    BuildComplexMatB(&data, opB);
    BuildComplexExpected(&data, opB);
    const bool setup = AllocateComplexDevice(data, &device) && UploadComplexInput(data, device) &&
        PrepareComplexScalars(data, deviceScalars, &device);
    const ComplexPrepareResult preparation = setup ?
        PrepareComplexSpmm(stream, data, opB, alg, deviceScalars, name, &device) :
        ComplexPrepareResult::kFailure;
    bool pass = preparation == ComplexPrepareResult::kExpectedRejection;
    if (preparation == ComplexPrepareResult::kReady) {
        pass = ExecuteComplexSpmm(stream, &data, opB, alg, deviceScalars, device) &&
            VerifyComplexResult(data, name);
    }
    CleanupComplexDevice(&device);
    return pass;
}

} // namespace

static bool RunTaskDtypeTests(aclrtStream stream)
{
    std::printf("##### A5_SPMM_TASK_DTYPE_START #####\n");
    bool pass = true;
    pass = RunRealCase(stream, "BF16_TO_BF16_ALG2", ACL_BF16, ACL_BF16,
                       ACL_SPARSE_SPMM_CSR_ALG2) && pass;
    pass = RunRealCase(stream, "BF16_TO_FP32_ALG1", ACL_BF16, ACL_FLOAT,
                       ACL_SPARSE_SPMM_CSR_ALG1) && pass;
    pass = RunRealCase(stream, "FP16_TO_FP32_ALG3", ACL_FLOAT16, ACL_FLOAT,
                       ACL_SPARSE_SPMM_CSR_ALG3) && pass;
    pass = RunComplexCase(stream, "COMPLEX64_N_HOST_ALG2", ACL_SPARSE_OP_NON_TRANSPOSE,
                          ACL_SPARSE_SPMM_CSR_ALG2, false) && pass;
    pass = RunComplexCase(stream, "COMPLEX64_H_DEVICE_ALG1", ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
                          ACL_SPARSE_SPMM_CSR_ALG1, true) && pass;
    pass = RunComplexCase(stream, "COMPLEX64_N_HOST_ALG3", ACL_SPARSE_OP_NON_TRANSPOSE,
                          ACL_SPARSE_SPMM_CSR_ALG3, false) && pass;
    pass = RunComplexCase(stream, "COMPLEX64_H_ALG3_REJECT", ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
                          ACL_SPARSE_SPMM_CSR_ALG3, false) && pass;
    std::printf("##### A5_SPMM_TASK_DTYPE_RESULT #####\n");
    std::printf("Overall: %s\n", pass ? "PASS" : "FAIL");

    return pass;
}

using spmm_test::SpmmDeviceTest;

TEST_F(SpmmDeviceTest, TaskDtypes)
{
    ASSERT_TRUE(RunTaskDtypeTests(stream_));
}
