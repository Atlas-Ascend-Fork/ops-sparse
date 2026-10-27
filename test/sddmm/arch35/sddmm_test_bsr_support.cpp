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
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace sparse_test {
namespace sddmm_extended {

void CheckComplexBsrL0CapacityBoundaries()
{
    struct CapacityCase {
        int32_t blockDim;
        int64_t reduction;
        uint64_t aBytes;
        uint64_t bBytes;
    };
    // Independent byte counts for all six supported shapes; B16/K128 must
    // cover the grouped path even though its single-block B tile is smaller.
    const CapacityCase cases[] = {
        {16, 64, 16384ULL, 8192ULL},
        {16, 128, 32768ULL, 65536ULL},
        {32, 64, 32768ULL, 16384ULL},
        {32, 128, 32768ULL, 16384ULL},
        {64, 64, 65536ULL, 32768ULL},
        {64, 128, 65536ULL, 32768ULL},
    };
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testing::Message() << testCase.blockDim << "/" << testCase.reduction);
        SddmmTilingData tiling{};
        tiling.blockDim = testCase.blockDim;
        tiling.k = testCase.reduction;
        const auto required = sddmm_arch35::SddmmComplexBsrCubeGetL0Requirements(tiling);
        EXPECT_EQ(required.aBytes, testCase.aBytes);
        EXPECT_EQ(required.bBytes, testCase.bBytes);
        EXPECT_TRUE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(
            tiling, testCase.aBytes, testCase.bBytes));
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(
            tiling, testCase.aBytes - 1ULL, testCase.bBytes));
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(
            tiling, testCase.aBytes, testCase.bBytes - 1ULL));
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling, 0ULL, testCase.bBytes));
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling, testCase.aBytes, 0ULL));
        EXPECT_TRUE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling,
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()));
    }
}

void CheckComplexBsrGroupedL0Requirement()
{
    using SingleTile = sddmm_arch35::SddmmComplexBsrCubeTileConfig<16, 128>;
    using GroupedTile = sddmm_arch35::SddmmComplexBsrCubeGroupedTileConfig;
    EXPECT_EQ(SingleTile::kABytes, 32768ULL);
    EXPECT_EQ(SingleTile::kBBytes, 16384ULL);
    EXPECT_EQ(GroupedTile::kABytes, 32768ULL);
    EXPECT_EQ(GroupedTile::kBBytes, 65536ULL);
    SddmmTilingData tiling{};
    tiling.blockDim = 16;
    tiling.k = 128;
    EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling, 32768ULL, 16384ULL));
    EXPECT_TRUE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling, 32768ULL, 65536ULL));
}

void CheckComplexBsrL0RejectsUnsupportedTiles()
{
    SddmmTilingData tiling{};
    for (const int32_t blockDim : {-1, 0, 8, 128}) {
        SCOPED_TRACE(blockDim);
        tiling.blockDim = blockDim;
        tiling.k = 128;
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling,
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()));
    }
    tiling.blockDim = 16;
    for (const int64_t reduction : {-1LL, 0LL, 32LL, 256LL, 4294967360LL}) {
        SCOPED_TRACE(reduction);
        tiling.k = reduction;
        EXPECT_FALSE(sddmm_arch35::SddmmComplexBsrCubeFitsL0(tiling,
            std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()));
    }
}

void CheckSetupStatus(aclsparseStatus_t status, const char *api)
{
    if (status != ACL_SPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(api) + " failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

size_t RealTypeBytes(aclDataType dataType)
{
    if (dataType == ACL_FLOAT) {
        return sizeof(float);
    }
    if (dataType == ACL_FLOAT16 || dataType == ACL_BF16) {
        return sizeof(uint16_t);
    }
    throw std::runtime_error("unsupported real SDDMM test dtype");
}

inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::copy_n(reinterpret_cast<const uint8_t *>(&value), sizeof(bits),
                reinterpret_cast<uint8_t *>(&bits));
    if ((bits & 0x7F800000U) == 0x7F800000U &&
        (bits & 0x007FFFFFU) != 0U) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    }
    const uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<uint16_t>((bits + 0x7FFFU + lsb) >> 16);
}

inline float Bf16BitsToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::copy_n(reinterpret_cast<const uint8_t *>(&bits), sizeof(result),
                reinterpret_cast<uint8_t *>(&result));
    return result;
}

float QuantizeReal(float value, aclDataType dataType)
{
    if (dataType == ACL_FLOAT) {
        return value;
    }
    if (dataType == ACL_FLOAT16) {
        return Fp16BitsToFp32(Fp32ToFp16Bits(value));
    }
    if (dataType == ACL_BF16) {
        return Bf16BitsToFloat(FloatToBf16Bits(value));
    }
    throw std::runtime_error("unsupported quantization dtype");
}

std::vector<float> QuantizeRealVector(
    const std::vector<float> &values, aclDataType dataType)
{
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = QuantizeReal(values[i], dataType);
    }
    return result;
}

template <typename T>
inline void StoreValue(std::vector<uint8_t> &bytes, size_t index, T value)
{
    if (index >= bytes.size() / sizeof(T)) {
        throw std::out_of_range("SDDMM test byte-store index");
    }
    std::copy_n(reinterpret_cast<const uint8_t *>(&value), sizeof(T),
                bytes.data() + index * sizeof(T));
}

template <typename T>
inline T LoadValue(const std::vector<uint8_t> &bytes, size_t index)
{
    T value{};
    if (index >= bytes.size() / sizeof(T)) {
        throw std::out_of_range("SDDMM test byte-load index");
    }
    std::copy_n(bytes.data() + index * sizeof(T), sizeof(T),
                reinterpret_cast<uint8_t *>(&value));
    return value;
}

std::vector<uint8_t> PackRealValues(
    const std::vector<float> &values, aclDataType dataType)
{
    std::vector<uint8_t> result(values.size() * RealTypeBytes(dataType));
    for (size_t i = 0; i < values.size(); ++i) {
        if (dataType == ACL_FLOAT) {
            StoreValue(result, i, values[i]);
        } else if (dataType == ACL_FLOAT16) {
            StoreValue(result, i, Fp32ToFp16Bits(values[i]));
        } else {
            StoreValue(result, i, FloatToBf16Bits(values[i]));
        }
    }
    return result;
}

std::vector<float> UnpackRealValues(
    const std::vector<uint8_t> &bytes, size_t count, aclDataType dataType)
{
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        if (dataType == ACL_FLOAT) {
            result[i] = LoadValue<float>(bytes, i);
        } else if (dataType == ACL_FLOAT16) {
            result[i] = Fp16BitsToFp32(LoadValue<uint16_t>(bytes, i));
        } else {
            result[i] = Bf16BitsToFloat(LoadValue<uint16_t>(bytes, i));
        }
    }
    return result;
}

std::vector<float> MakeRealValues(
    int batchCount, int64_t elementsPerBatch, float phase)
{
    std::vector<float> result(
        static_cast<size_t>(batchCount) * static_cast<size_t>(elementsPerBatch));
    for (int batch = 0; batch < batchCount; ++batch) {
        for (int64_t i = 0; i < elementsPerBatch; ++i) {
            const int term = static_cast<int>((i * 7 + batch * 5) % 17) - 8;
            result[static_cast<size_t>(batch) *
                       static_cast<size_t>(elementsPerBatch) +
                   static_cast<size_t>(i)] =
                phase + static_cast<float>(term) * 0.03125F +
                static_cast<float>(batch) * 0.125F;
        }
    }
    return result;
}

inline std::vector<float> MakeRandomTaskValues(
    size_t count, uint32_t seed, aclDataType dataType)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const float epsilon = dataType == ACL_FLOAT16 ? 0x1p-10F :
        dataType == ACL_BF16 ? 0x1p-7F : 0x1p-23F;
    const float tiny = dataType == ACL_FLOAT16 ? 0x1p-14F : 0x1p-126F;
    const float subnormal = dataType == ACL_FLOAT16 ? 0x1p-24F :
        dataType == ACL_BF16 ? 0x1p-133F : 0x1p-149F;
    const float boundary[] = {0.0F, epsilon, -epsilon, tiny, -tiny,
                              8.0F, -8.0F, 64.0F, -64.0F,
                              subnormal, -subnormal};
    const size_t uniformCount = count * 7U / 10U;
    const size_t normalCount = count * 2U / 10U;
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = i < uniformCount ? uniform(generator) :
            i < uniformCount + normalCount ? normal(generator) :
            boundary[(i - uniformCount - normalCount) % 11U];
    }
    std::shuffle(result.begin(), result.end(), generator);
    return QuantizeRealVector(result, dataType);
}

void CheckGuardedBuffers(
    const SddmmGuardedBuffer &values, const SddmmGuardedBuffer &workspace,
    const std::vector<const SddmmGuardedBuffer *> &inputs)
{
    EXPECT_TRUE(values.guardsIntact()) << "output canary";
    EXPECT_TRUE(workspace.guardsIntact()) << "workspace canary";
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_TRUE(inputs[i]->unchanged()) << "readonly input " << i;
        EXPECT_TRUE(inputs[i]->guardsIntact()) << "input canary " << i;
    }
}

inline void RepeatGuardedExecute(
    SddmmStageResult &result, HandleManager &handle, const void *alpha, const void *beta,
    DnMatManager &matX, DnMatManager &matY, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclrtStream stream, SddmmGuardedBuffer &values,
    SddmmGuardedBuffer &workspace, const std::vector<const SddmmGuardedBuffer *> &inputs)
{
    std::vector<uint8_t> first;
    for (int repetition = 0; repetition < 2; ++repetition) {
        SCOPED_TRACE("repetition=" + std::to_string(repetition));
        if (repetition != 0) {
            values.restore();
        }
        result.executeRet = aclsparseSDDMM(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
            ACL_SPARSE_OP_NON_TRANSPOSE, alpha, matX.cget(), matY.cget(),
            beta, matC, computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT,
            workspace.get());
        if (result.executeRet != ACL_SPARSE_STATUS_SUCCESS) {
            return;
        }
        result.synchronizeRet = aclrtSynchronizeStream(stream);
        if (result.synchronizeRet != ACL_SUCCESS) {
            return;
        }
        CheckGuardedBuffers(values, workspace, inputs);
        if (repetition == 0) {
            first = values.snapshot();
        } else {
            EXPECT_EQ(first, values.snapshot()) << "repeat bitwise after restoring C";
        }
    }
}

void SetHostTestStream(HandleManager &handle, aclrtStream stream)
{
    handle.setStream(stream);
    CheckSetupStatus(aclsparseSetPointerMode(
        handle.get(), ACL_SPARSE_POINTER_MODE_HOST), "aclsparseSetPointerMode");
}

bool QueryTestWorkspace(SddmmStageResult &result, HandleManager &handle,
    const void *alpha, const void *beta, DnMatManager &matX, DnMatManager &matY,
    aclsparseSpMatDescr_t matC, aclDataType computeType)
{
    result.bufferSizeRet = aclsparseSDDMMBufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, matX.cget(), matY.cget(),
        beta, matC, computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        &result.workspaceSize);
    result.workspaceSizeAligned = result.workspaceSize == 0 ||
        result.workspaceSize % kSddmmWorkspaceAlignment == 0;
    return result.bufferSizeRet == ACL_SPARSE_STATUS_SUCCESS;
}

inline SddmmStageResult RunGuardedRepeatedStages(
    HandleManager &handle, const void *alpha, const void *beta,
    DnMatManager &matX, DnMatManager &matY, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclrtStream stream, SddmmGuardedBuffer &values,
    const std::vector<const SddmmGuardedBuffer *> &inputs)
{
    SddmmStageResult result;
    if (!QueryTestWorkspace(result, handle, alpha, beta, matX, matY, matC, computeType)) {
        return result;
    }
    auto workspace = SddmmGuardedBuffer::copyFrom(nullptr, result.workspaceSize);
    result.workspaceSizeAligned = result.workspaceSize == 0 ||
        result.workspaceSize % kSddmmWorkspaceAlignment == 0;
    result.workspacePointerAligned = workspace.get() == nullptr ||
        reinterpret_cast<uintptr_t>(workspace.get()) % 64U == 0;
    result.preprocessRet = aclsparseSDDMMPreprocess(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, matX.cget(), matY.cget(),
        beta, matC, computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    if (result.preprocessRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    result.synchronizeRet = aclrtSynchronizeStream(stream);
    if (result.synchronizeRet != ACL_SUCCESS) {
        return result;
    }
    EXPECT_TRUE(values.unchanged()) << "Preprocess must preserve C values";
    CheckGuardedBuffers(values, workspace, inputs);
    RepeatGuardedExecute(result, handle, alpha, beta, matX, matY, matC,
        computeType, stream, values, workspace, inputs);
    return result;
}

SddmmStageResult RunThreeStages(
    HandleManager &handle, const void *alpha, const void *beta,
    DnMatManager &matX, DnMatManager &matY, aclsparseSpMatDescr_t matC,
    aclDataType computeType, aclrtStream stream, DeviceBuffer &workspace)
{
    SddmmStageResult result;
    if (!QueryTestWorkspace(result, handle, alpha, beta, matX, matY, matC, computeType)) {
        return result;
    }
    result.workspaceSizeAligned = result.workspaceSize == 0 ||
        result.workspaceSize % kSddmmWorkspaceAlignment == 0;
    if (result.workspaceSize > 0) {
        workspace = DeviceBuffer::alloc(result.workspaceSize);
        result.workspacePointerAligned =
            reinterpret_cast<uintptr_t>(workspace.get()) %
                kSddmmWorkspaceAlignment == 0;
    }
    result.preprocessRet = aclsparseSDDMMPreprocess(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, matX.cget(), matY.cget(),
        beta, matC, computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    if (result.preprocessRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    result.executeRet = aclsparseSDDMM(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, alpha, matX.cget(), matY.cget(),
        beta, matC, computeType, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    if (result.executeRet == ACL_SPARSE_STATUS_SUCCESS) {
        result.synchronizeRet = aclrtSynchronizeStream(stream);
    }
    return result;
}

template <typename Case>
inline void CreateBsrTestDescriptor(BsrMatGuard &matC, const Case &testCase,
    aclDataType outputType, void *offsets, void *columns, void *values)
{
    CheckSetupStatus(aclsparseCreateBsr(
        matC.put(), testCase.blockRows, testCase.blockCols,
        static_cast<int64_t>(testCase.zeroBasedColumns.size()),
        testCase.blockSize, testCase.blockSize, offsets, columns, values,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        testCase.indexBase, outputType, testCase.blockOrder), "aclsparseCreateBsr");
}

inline int64_t FindBlockRow(
    const std::vector<int32_t> &offsets, int64_t blockSlot)
{
    int64_t blockRow = 0;
    while (blockRow + 1 < static_cast<int64_t>(offsets.size()) &&
           blockSlot >= offsets[static_cast<size_t>(blockRow + 1)]) {
        ++blockRow;
    }
    return blockRow;
}

float RealDotGolden(const float *x, const float *y, int64_t k,
                           int64_t yStride, aclDataType inputType,
                           float alpha, float beta, float oldValue)
{
    if (alpha == 0.0F) {
        return beta == 0.0F ? 0.0F : beta * oldValue;
    }
    if (inputType == ACL_FLOAT) {
        double dot = 0.0;
        for (int64_t reduction = 0; reduction < k; ++reduction) {
            dot = std::fma(static_cast<double>(x[reduction]),
                static_cast<double>(y[reduction * yStride]), dot);
        }
        return static_cast<float>(static_cast<double>(alpha) * dot +
            static_cast<double>(beta) * oldValue);
    }
    float dot = 0.0F;
    for (int64_t reduction = 0; reduction < k; ++reduction) {
        dot = std::fma(x[reduction], y[reduction * yStride], dot);
    }
    return std::fma(alpha, dot, beta * oldValue);
}

std::vector<float> BsrRealGolden(
    const BsrRealCase &testCase, const std::vector<float> &x,
    const std::vector<float> &y, const std::vector<float> &cInitial)
{
    const int64_t m = testCase.blockRows * testCase.blockSize;
    const int64_t n = testCase.blockCols * testCase.blockSize;
    const int64_t blockArea = testCase.blockSize * testCase.blockSize;
    const int64_t outputsPerBatch =
        static_cast<int64_t>(testCase.zeroBasedColumns.size()) * blockArea;
    const int64_t xStride = m * testCase.k;
    const int64_t yStride = testCase.k * n;
    std::vector<float> result(
        static_cast<size_t>(testCase.outputBatchCount) *
        static_cast<size_t>(outputsPerBatch));

    for (int batch = 0; batch < testCase.outputBatchCount; ++batch) {
        const int xBatch = testCase.xBatchCount == 1 ? 0 : batch;
        const int yBatch = testCase.yBatchCount == 1 ? 0 : batch;
        for (int64_t localOutput = 0;
             localOutput < outputsPerBatch; ++localOutput) {
            const int64_t blockSlot = localOutput / blockArea;
            const int64_t inner = localOutput % blockArea;
            const int64_t blockRow =
                FindBlockRow(testCase.zeroBasedOffsets, blockSlot);
            const int64_t blockColumn = testCase.zeroBasedColumns[
                static_cast<size_t>(blockSlot)];
            const int64_t innerRow = testCase.blockOrder == ACL_SPARSE_ORDER_ROW ?
                inner / testCase.blockSize : inner % testCase.blockSize;
            const int64_t innerColumn = testCase.blockOrder == ACL_SPARSE_ORDER_ROW ?
                inner % testCase.blockSize : inner / testCase.blockSize;
            const int64_t row = blockRow * testCase.blockSize + innerRow;
            const int64_t column =
                blockColumn * testCase.blockSize + innerColumn;

            const size_t outputIndex =
                static_cast<size_t>(batch * outputsPerBatch + localOutput);
            const float value = RealDotGolden(
                x.data() + xBatch * xStride + row * testCase.k,
                y.data() + yBatch * yStride + column, testCase.k, n,
                testCase.inputType, testCase.alpha, testCase.beta,
                cInitial[outputIndex]);
            result[outputIndex] = QuantizeReal(value, testCase.outputType);
        }
    }
    return result;
}

class BsrRealCaseRunner : private BsrTestResources {
public:
    BsrRealCaseRunner(aclrtStream callStream, const BsrRealCase &params)
        : testCase(params), stream(callStream) {}

    RealRunResult Run()
    {
        PreparePattern();
        PrepareDenseValues();
        PrepareOutputValues();
        CreateDescriptors();
        PrepareScalars();
        return Execute();
    }

private:
    const BsrRealCase &testCase;
    aclrtStream stream;
    int64_t m{};
    int64_t n{};
    int64_t blockNnz{};
    int64_t valuesPerBatch{};
    int64_t xStride{};
    int64_t yStride{};
    std::vector<int32_t> offsets{};
    std::vector<int32_t> columns{};
    std::vector<float> x{};
    std::vector<float> y{};
    std::vector<float> cInitial{};
    std::vector<uint8_t> xBytes{};
    std::vector<uint8_t> yBytes{};
    std::vector<uint8_t> cBytes{};

    void PreparePattern()
    {
        m = testCase.blockRows * testCase.blockSize;
        n = testCase.blockCols * testCase.blockSize;
        blockNnz = static_cast<int64_t>(testCase.zeroBasedColumns.size());
        valuesPerBatch = blockNnz * testCase.blockSize * testCase.blockSize;
        xStride = m * testCase.k;
        yStride = testCase.k * n;

        offsets = testCase.zeroBasedOffsets;
        columns = testCase.zeroBasedColumns;
        const int32_t base = static_cast<int32_t>(testCase.indexBase);
        for (auto &value : offsets) {
            value += base;
        }
        for (auto &value : columns) {
            value += base;
        }
    }

    void PrepareDenseValues()
    {
        x = testCase.randomValues ? MakeRandomTaskValues(
            static_cast<size_t>(testCase.xBatchCount * xStride), 26091201U,
            testCase.inputType) : QuantizeRealVector(
            MakeRealValues(testCase.xBatchCount, xStride, 0.125F),
            testCase.inputType);
        y = testCase.randomValues ? MakeRandomTaskValues(
            static_cast<size_t>(testCase.yBatchCount * yStride), 26091202U,
            testCase.inputType) : QuantizeRealVector(
            MakeRealValues(testCase.yBatchCount, yStride, -0.0625F),
            testCase.inputType);
        if (testCase.specialDenseValues) {
            std::fill(x.begin(), x.end(),
                std::numeric_limits<float>::quiet_NaN());
            std::fill(y.begin(), y.end(),
                std::numeric_limits<float>::infinity());
        }
        if (testCase.subnormalDenseValues) {
            if (testCase.inputType != ACL_FLOAT16) {
                throw std::runtime_error("subnormal load case requires FP16 input");
            }
            for (size_t i = 0; i < x.size(); ++i) {
                const float value = Fp16BitsToFp32(static_cast<uint16_t>(1U + i % 7U));
                x[i] = (i / static_cast<size_t>(testCase.k)) % 2U == 0 ? value : -value;
            }
            std::fill(y.begin(), y.end(), 32768.0F);
        }
    }

    void PrepareOutputValues()
    {
        cInitial = !testCase.initialValuesOverride.empty() ?
            testCase.initialValuesOverride : (testCase.randomValues ? MakeRandomTaskValues(
            static_cast<size_t>(testCase.outputBatchCount * valuesPerBatch),
            26091203U, testCase.outputType) : QuantizeRealVector(
            MakeRealValues(testCase.outputBatchCount, valuesPerBatch, 0.03125F),
            testCase.outputType));
        if (cInitial.size() != static_cast<size_t>(testCase.outputBatchCount * valuesPerBatch)) {
            throw std::runtime_error("BSR initial-values override has the wrong size");
        }
        xBytes = PackRealValues(x, testCase.inputType);
        yBytes = PackRealValues(y, testCase.inputType);
        cBytes = PackRealValues(cInitial, testCase.outputType);
    }

    void CreateDescriptors()
    {
        dOffsets = SddmmGuardedBuffer::copyFrom(
            offsets.data(), offsets.size() * sizeof(int32_t));
        dColumns = SddmmGuardedBuffer::copyFrom(
            columns.data(), columns.size() * sizeof(int32_t));
        dX = SddmmGuardedBuffer::copyFrom(xBytes.data(), xBytes.size());
        dY = SddmmGuardedBuffer::copyFrom(yBytes.data(), yBytes.size());
        dValues = SddmmGuardedBuffer::copyFrom(cBytes.data(), cBytes.size());

        matX = DnMatManager::create(
            m, testCase.k, testCase.k, dX.get(), testCase.inputType,
            ACL_SPARSE_ORDER_ROW);
        matY = DnMatManager::create(
            testCase.k, n, n, dY.get(), testCase.inputType,
            ACL_SPARSE_ORDER_ROW);
        CheckSetupStatus(aclsparseDnMatSetStridedBatch(
                             matX.get(), testCase.xBatchCount,
                             testCase.xBatchCount == 1 ? 0 : xStride),
                         "aclsparseDnMatSetStridedBatch(X)");
        CheckSetupStatus(aclsparseDnMatSetStridedBatch(
                             matY.get(), testCase.yBatchCount,
                             testCase.yBatchCount == 1 ? 0 : yStride),
                         "aclsparseDnMatSetStridedBatch(Y)");

        CreateBsrTestDescriptor(matC, testCase, testCase.outputType,
            dOffsets.get(), dColumns.get(), dValues.get());
        CheckSetupStatus(aclsparseBsrSetStridedBatch(
                             matC.get(), testCase.outputBatchCount, 0, 0,
                             testCase.outputBatchCount == 1 ? 0 : valuesPerBatch),
                         "aclsparseBsrSetStridedBatch");
    }

    void PrepareScalars()
    {
        SetScalars(stream, testCase.deviceScalars, testCase.alpha, testCase.beta);
    }

    RealRunResult Execute()
    {
        RealRunResult result;
        result.valuesPointer64Aligned =
            (reinterpret_cast<uintptr_t>(dValues.get()) & 63U) == 0U;
        result.stages = testCase.guardedRepeat ? RunGuardedRepeatedStages(
            handle, alpha, beta, matX, matY, matC.get(), ACL_FLOAT, stream,
            dValues, {&dOffsets, &dColumns, &dX, &dY}) : RunThreeStages(
            handle, alpha, beta, matX, matY, matC.get(), ACL_FLOAT, stream,
            workspace);
        if (result.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS ||
            result.stages.synchronizeRet != ACL_SUCCESS) {
            return result;
        }

        std::vector<uint8_t> outputBytes(cBytes.size());
        dValues.copyToHost(outputBytes.data(), outputBytes.size());
        result.actual = UnpackRealValues(
            outputBytes,
            static_cast<size_t>(testCase.outputBatchCount * valuesPerBatch),
            testCase.outputType);
        result.golden = BsrRealGolden(testCase, x, y, cInitial);
        return result;
    }
};

RealRunResult RunBsrRealCase(aclrtStream stream, const BsrRealCase &testCase)
{
    BsrRealCaseRunner runner(stream, testCase);
    return runner.Run();
}

bool VerifyRealRun(
    const RealRunResult &result, aclDataType outputType,
    const std::string &caseName)
{
    if (result.actual.size() != result.golden.size()) {
        return false;
    }
    VerifyConfig config;
    applyMixedTolerance(
        config, outputType, result.golden.data(), result.golden.size());
    return Verifier::verifyVector(
        result.actual, result.golden, config, caseName);
}

std::vector<aclsparseComplex> MakeComplexValues(
    size_t count, float phase)
{
    std::vector<aclsparseComplex> result(count);
    for (size_t i = 0; i < count; ++i) {
        const int realTerm = static_cast<int>((i * 3) % 11) - 5;
        const int imagTerm = static_cast<int>((i * 5) % 13) - 6;
        result[i].x = phase + static_cast<float>(realTerm) * 0.0625F;
        result[i].y = -phase + static_cast<float>(imagTerm) * 0.03125F;
    }
    return result;
}

inline std::vector<aclsparseComplex> MakeRandomTaskComplexValues(
    size_t count, uint32_t seed)
{
    const auto real = MakeRandomTaskValues(count, seed, ACL_FLOAT);
    const auto imag = MakeRandomTaskValues(count, seed + 1U, ACL_FLOAT);
    std::vector<aclsparseComplex> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = {real[i], imag[i]};
    }
    return result;
}

inline std::complex<double> ToComplex128(const aclsparseComplex &value)
{
    return {static_cast<double>(value.x), static_cast<double>(value.y)};
}

std::vector<float> BsrComplexGoldenParts(
    const BsrComplexCase &testCase,
    const std::vector<aclsparseComplex> &x,
    const std::vector<aclsparseComplex> &y,
    const std::vector<aclsparseComplex> &cInitial)
{
    const int64_t n = testCase.blockCols * testCase.blockSize;
    const int64_t blockArea = testCase.blockSize * testCase.blockSize;
    const int64_t outputCount =
        static_cast<int64_t>(testCase.zeroBasedColumns.size()) * blockArea;
    std::vector<float> result(static_cast<size_t>(outputCount) * 2U);
    const auto alpha = ToComplex128(testCase.alpha);
    const auto beta = ToComplex128(testCase.beta);

    for (int64_t output = 0; output < outputCount; ++output) {
        const int64_t blockSlot = output / blockArea;
        const int64_t inner = output % blockArea;
        const int64_t blockRow = FindBlockRow(
            testCase.zeroBasedOffsets, blockSlot);
        const int64_t blockColumn = testCase.zeroBasedColumns[
            static_cast<size_t>(blockSlot)];
        const int64_t innerRow =
            testCase.blockOrder == ACL_SPARSE_ORDER_COL ?
                inner % testCase.blockSize : inner / testCase.blockSize;
        const int64_t innerColumn =
            testCase.blockOrder == ACL_SPARSE_ORDER_COL ?
                inner / testCase.blockSize : inner % testCase.blockSize;
        const int64_t row = blockRow * testCase.blockSize +
            innerRow;
        const int64_t column = blockColumn * testCase.blockSize +
            innerColumn;
        std::complex<double> dot{0.0, 0.0};
        for (int64_t reduction = 0;
             alpha != std::complex<double>{0.0, 0.0} &&
             reduction < testCase.k; ++reduction) {
            dot += ToComplex128(x[static_cast<size_t>(
                       row * testCase.k + reduction)]) *
                   ToComplex128(y[static_cast<size_t>(
                       reduction * n + column)]);
        }
        const std::complex<double> expected = alpha * dot +
            (beta == std::complex<double>{0.0, 0.0} ?
                std::complex<double>{0.0, 0.0} :
                beta * ToComplex128(cInitial[static_cast<size_t>(output)]));
        result[static_cast<size_t>(output) * 2U] =
            static_cast<float>(expected.real());
        result[static_cast<size_t>(output) * 2U + 1U] =
            static_cast<float>(expected.imag());
    }
    return result;
}

class BsrComplexCaseRunner : private BsrTestResources {
public:
    BsrComplexCaseRunner(aclrtStream callStream, const BsrComplexCase &params)
        : testCase(params), stream(callStream) {}

    ComplexRunResult Run()
    {
        PrepareInputs();
        CreateDescriptors();
        PrepareScalars();
        return Execute();
    }

private:
    const BsrComplexCase &testCase;
    aclrtStream stream;
    int64_t m{};
    int64_t n{};
    int64_t blockNnz{};
    int64_t outputCount{};
    std::vector<aclsparseComplex> x{};
    std::vector<aclsparseComplex> y{};
    std::vector<aclsparseComplex> cInitial{};
    std::vector<int32_t> offsets{};
    std::vector<int32_t> columns{};

    void PrepareInputs()
    {
        m = testCase.blockRows * testCase.blockSize;
        n = testCase.blockCols * testCase.blockSize;
        blockNnz = static_cast<int64_t>(
            testCase.zeroBasedColumns.size());
        outputCount = blockNnz * testCase.blockSize * testCase.blockSize;
        x = testCase.randomValues ? MakeRandomTaskComplexValues(
            static_cast<size_t>(m * testCase.k), 26091211U) : MakeComplexValues(
            static_cast<size_t>(m * testCase.k), 0.125F);
        y = testCase.randomValues ? MakeRandomTaskComplexValues(
            static_cast<size_t>(testCase.k * n), 26091213U) : MakeComplexValues(
            static_cast<size_t>(testCase.k * n), -0.0625F);
        cInitial = testCase.randomValues ? MakeRandomTaskComplexValues(
            static_cast<size_t>(outputCount), 26091215U) : MakeComplexValues(
            static_cast<size_t>(outputCount), 0.03125F);
        if (testCase.specialDenseValues) {
            std::fill(x.begin(), x.end(), aclsparseComplex{
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()});
            std::fill(y.begin(), y.end(), aclsparseComplex{
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity()});
        }
        offsets = testCase.zeroBasedOffsets;
        columns = testCase.zeroBasedColumns;
        const int32_t base = static_cast<int32_t>(testCase.indexBase);
        for (auto &offset : offsets) {
            offset += base;
        }
        for (auto &column : columns) {
            column += base;
        }
    }

    void CreateDescriptors()
    {
        dOffsets = SddmmGuardedBuffer::copyFrom(
            offsets.data(), offsets.size() * sizeof(int32_t));
        dColumns = SddmmGuardedBuffer::copyFrom(
            columns.data(), columns.size() * sizeof(int32_t));
        const size_t xBytes = x.size() * sizeof(aclsparseComplex);
        const size_t yBytes = y.size() * sizeof(aclsparseComplex);
        const size_t denseOffset = testCase.offsetDensePointersByOneElement ?
            sizeof(aclsparseComplex) : 0U;
        dX = SddmmGuardedBuffer::copyFrom(x.data(), xBytes, denseOffset);
        std::vector<aclsparseComplex> columnY;
        if (testCase.columnMajorY) {
            columnY.resize(y.size());
            for (int64_t reduction = 0; reduction < testCase.k; ++reduction) {
                for (int64_t column = 0; column < n; ++column) {
                    columnY[static_cast<size_t>(column * testCase.k + reduction)] =
                        y[static_cast<size_t>(reduction * n + column)];
                }
            }
        }
        dY = SddmmGuardedBuffer::copyFrom(
            testCase.columnMajorY ? columnY.data() : y.data(), yBytes, denseOffset);
        const void *xDevice = dX.get();
        const void *yDevice = dY.get();
        dValues = SddmmGuardedBuffer::copyFrom(
            cInitial.data(), cInitial.size() * sizeof(aclsparseComplex));
        matX = DnMatManager::createConst(
            m, testCase.k, testCase.k, xDevice, ACL_COMPLEX64,
            ACL_SPARSE_ORDER_ROW);
        matY = DnMatManager::createConst(
            testCase.k, n, testCase.columnMajorY ? testCase.k : n,
            yDevice, ACL_COMPLEX64,
            testCase.columnMajorY ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW);
        CreateBsrTestDescriptor(matC, testCase, ACL_COMPLEX64,
            dOffsets.get(), dColumns.get(), dValues.get());
    }

    void PrepareScalars()
    {
        SetScalars(stream, testCase.deviceScalars, testCase.alpha, testCase.beta);
    }

    ComplexRunResult Execute()
    {
        ComplexRunResult result;
        result.stages = testCase.guardedRepeat ? RunGuardedRepeatedStages(
            handle, alpha, beta, matX, matY, matC.get(),
            ACL_COMPLEX64, stream, dValues, {&dOffsets, &dColumns, &dX, &dY}) :
            RunThreeStages(
            handle, alpha, beta, matX, matY, matC.get(),
            ACL_COMPLEX64, stream, workspace);
        if (result.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS ||
            result.stages.synchronizeRet != ACL_SUCCESS) {
            return result;
        }

        std::vector<aclsparseComplex> actual(static_cast<size_t>(outputCount));
        dValues.copyToHost(
            actual.data(), actual.size() * sizeof(aclsparseComplex));
        result.actualParts.resize(actual.size() * 2U);
        for (size_t i = 0; i < actual.size(); ++i) {
            result.actualParts[i * 2U] = actual[i].x;
            result.actualParts[i * 2U + 1U] = actual[i].y;
        }
        result.goldenParts = BsrComplexGoldenParts(
            testCase, x, y, cInitial);
        return result;
    }
};

ComplexRunResult RunBsrComplexCase(aclrtStream stream, const BsrComplexCase &testCase)
{
    BsrComplexCaseRunner runner(stream, testCase);
    return runner.Run();
}

inline void FillBaseOneComplexResults(ComplexRunResult &result,
    const std::vector<aclsparseComplex> &actual,
    const std::vector<aclsparseComplex> &x,
    const std::vector<aclsparseComplex> &y,
    const std::vector<aclsparseComplex> &cInitial,
    const std::vector<int32_t> &rowOffsets, const std::vector<int32_t> &columns,
    int64_t m, int64_t n, int64_t k,
    const aclsparseComplex &alpha, const aclsparseComplex &beta)
{
    const int64_t nnz = static_cast<int64_t>(columns.size());
    result.actualParts.resize(static_cast<size_t>(nnz) * 2);
    result.goldenParts.resize(static_cast<size_t>(nnz) * 2);
    for (int64_t row = 0; row < m; ++row) {
        const int64_t begin = rowOffsets[static_cast<size_t>(row)] - 1;
        const int64_t end = rowOffsets[static_cast<size_t>(row + 1)] - 1;
        for (int64_t position = begin; position < end; ++position) {
            const int64_t column = columns[static_cast<size_t>(position)] - 1;
            std::complex<double> dot(0.0, 0.0);
            for (int64_t reduction = 0; reduction < k; ++reduction) {
                const auto &xValue = x[static_cast<size_t>(row * k + reduction)];
                const auto &yValue = y[static_cast<size_t>(reduction * n + column)];
                dot += std::complex<double>(xValue.x, xValue.y) *
                       std::complex<double>(yValue.x, yValue.y);
            }
            const auto &oldValue = cInitial[static_cast<size_t>(position)];
            const std::complex<double> expected =
                std::complex<double>(alpha.x, alpha.y) * dot +
                std::complex<double>(beta.x, beta.y) *
                    std::complex<double>(oldValue.x, oldValue.y);
            const size_t part = static_cast<size_t>(position) * 2;
            result.actualParts[part] = actual[static_cast<size_t>(position)].x;
            result.actualParts[part + 1] = actual[static_cast<size_t>(position)].y;
            result.goldenParts[part] = expected.real();
            result.goldenParts[part + 1] = expected.imag();
        }
    }
}

ComplexRunResult RunCsrBaseOneComplex(aclrtStream stream)
{
    constexpr int64_t m = 3;
    constexpr int64_t n = 4;
    constexpr int64_t k = 3;
    const std::vector<int32_t> rowOffsets = {1, 3, 4, 6};
    const std::vector<int32_t> columns = {1, 4, 2, 1, 3};
    const int64_t nnz = static_cast<int64_t>(columns.size());
    const auto x = MakeComplexValues(static_cast<size_t>(m * k), 0.125F);
    const auto y = MakeComplexValues(static_cast<size_t>(k * n), -0.0625F);
    const auto cInitial = MakeComplexValues(static_cast<size_t>(nnz), 0.03125F);
    const aclsparseComplex alpha{0.75F, -0.25F};
    const aclsparseComplex beta{-0.5F, 0.125F};

    DeviceBuffer dOffsets = DeviceBuffer::copyFrom(
        rowOffsets.data(), rowOffsets.size() * sizeof(int32_t));
    DeviceBuffer dColumns = DeviceBuffer::copyFrom(
        columns.data(), columns.size() * sizeof(int32_t));
    DeviceBuffer dX = DeviceBuffer::copyFrom(
        x.data(), x.size() * sizeof(aclsparseComplex));
    DeviceBuffer dY = DeviceBuffer::copyFrom(
        y.data(), y.size() * sizeof(aclsparseComplex));
    DeviceBuffer dValues = DeviceBuffer::copyFrom(
        cInitial.data(), cInitial.size() * sizeof(aclsparseComplex));
    auto matX = DnMatManager::createConst(
        m, k, k, dX.get(), ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(
        k, n, n, dY.get(), ACL_COMPLEX64, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(
        m, n, nnz, dOffsets.get(), dColumns.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ONE, ACL_COMPLEX64);

    HandleManager handle;
    SetHostTestStream(handle, stream);
    DeviceBuffer workspace;
    ComplexRunResult result;
    result.stages = RunThreeStages(
        handle, &alpha, &beta, matX, matY, matC.get(), ACL_COMPLEX64,
        stream, workspace);
    if (result.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS ||
        result.stages.synchronizeRet != ACL_SUCCESS) {
        return result;
    }

    std::vector<aclsparseComplex> actual(static_cast<size_t>(nnz));
    dValues.copyToHost(actual.data(), actual.size() * sizeof(aclsparseComplex));
    FillBaseOneComplexResults(result, actual, x, y, cInitial,
        rowOffsets, columns, m, n, k, alpha, beta);
    return result;
}

bool VerifyComplexRun(
    const ComplexRunResult &result, const std::string &caseName)
{
    if (result.actualParts.size() != result.goldenParts.size()) {
        return false;
    }
    VerifyConfig config;
    applyMixedTolerance(
        config, ACL_FLOAT, result.goldenParts.data(),
        result.goldenParts.size());
    return Verifier::verifyVector(
        result.actualParts, result.goldenParts, config, caseName);
}

bool StagesSucceeded(const SddmmStageResult &stages)
{
    return stages.bufferSizeRet == ACL_SPARSE_STATUS_SUCCESS &&
           stages.workspaceSizeAligned && stages.workspacePointerAligned &&
           stages.preprocessRet == ACL_SPARSE_STATUS_SUCCESS &&
           stages.executeRet == ACL_SPARSE_STATUS_SUCCESS &&
           stages.synchronizeRet == ACL_SUCCESS;
}

void AddComplexCubeBlock16Cases(std::vector<BsrComplexCase> &cases)
{
    BsrComplexCase block16;
    block16.name = "BSR_COMPLEX64_CUBE_B16_BASE0_COLUMN_WRAP";
    block16.blockCols = 4;
    block16.zeroBasedColumns = {3, 0, 2};

    BsrComplexCase block16CopyEpilogue = block16;
    block16CopyEpilogue.name =
        "BSR_COMPLEX64_CUBE_B16_COPY_TILE_BOUNDARY_TAIL";
    // The copy epilogue processes 4096 scalar components per tile: 16 BSR
    // blocks for b16.  Seventeen blocks cover one full tile and a one-block
    // tail while retaining a wrapped column order.
    block16CopyEpilogue.blockRows = 1;
    block16CopyEpilogue.blockCols = 17;
    block16CopyEpilogue.zeroBasedOffsets = {0, 17};
    block16CopyEpilogue.zeroBasedColumns = {
        16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    block16CopyEpilogue.alpha = {1.0F, 0.0F};
    block16CopyEpilogue.beta = {0.0F, 0.0F};

    cases.push_back(block16);
    cases.push_back(block16CopyEpilogue);
}

void AddComplexCubeBlock32Cases(std::vector<BsrComplexCase> &cases)
{
    BsrComplexCase block32;
    block32.name = "BSR_COMPLEX64_CUBE_B32_BASE1";
    block32.blockSize = 32;
    block32.zeroBasedOffsets = {0, 1, 3};
    block32.zeroBasedColumns = {0, 2, 1};
    block32.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    block32.alpha = {-0.625F, 0.375F};
    block32.beta = {0.25F, -0.75F};

    BsrComplexCase block32CopyEpilogue = block32;
    block32CopyEpilogue.name =
        "BSR_COMPLEX64_CUBE_B32_COPY_TILE_BOUNDARY_TAIL";
    // b32 has four blocks per copy tile.  Five blocks exercise the first
    // tile boundary and the one-block tail using base zero.
    block32CopyEpilogue.blockRows = 1;
    block32CopyEpilogue.blockCols = 5;
    block32CopyEpilogue.zeroBasedOffsets = {0, 5};
    block32CopyEpilogue.zeroBasedColumns = {4, 0, 1, 2, 3};
    block32CopyEpilogue.indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    block32CopyEpilogue.alpha = {1.0F, 0.0F};
    block32CopyEpilogue.beta = {0.0F, 0.0F};

    cases.push_back(block32);
    cases.push_back(block32CopyEpilogue);
}

void AddComplexCubeBlock64Cases(std::vector<BsrComplexCase> &cases)
{
    BsrComplexCase block64;
    block64.name = "BSR_COMPLEX64_CUBE_B64_BASE1";
    block64.blockRows = 1;
    block64.blockCols = 2;
    block64.blockSize = 64;
    block64.zeroBasedOffsets = {0, 2};
    block64.zeroBasedColumns = {1, 0};
    block64.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    block64.alpha = {0.5F, 0.625F};
    block64.beta = {-0.375F, -0.25F};

    BsrComplexCase block64CopyEpilogue = block64;
    block64CopyEpilogue.name =
        "BSR_COMPLEX64_CUBE_B64_COPY_TILE_BOUNDARIES";
    // A b64 block fills one 4096-component copy tile.  Three blocks cross two
    // independent tile boundaries (there is no fractional-block b64 tail).
    block64CopyEpilogue.blockRows = 1;
    block64CopyEpilogue.blockCols = 3;
    block64CopyEpilogue.zeroBasedOffsets = {0, 3};
    block64CopyEpilogue.zeroBasedColumns = {2, 0, 1};
    block64CopyEpilogue.indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    block64CopyEpilogue.alpha = {1.0F, 0.0F};
    block64CopyEpilogue.beta = {0.0F, 0.0F};

    cases.push_back(block64);
    cases.push_back(block64CopyEpilogue);
}

BsrComplexCase MakeComplexColumnIdentityCase(int64_t reduction)
{
    BsrComplexCase testCase;
    testCase.name = "BSR_COMPLEX64_B16_COL_IDENTITY_MULTIPLE_TILES_TAIL";
    testCase.blockRows = 70;
    testCase.blockCols = 37;
    testCase.k = reduction;
    testCase.indexBase = reduction == 64 ? ACL_SPARSE_INDEX_BASE_ZERO :
        ACL_SPARSE_INDEX_BASE_ONE;
    testCase.alpha = {1.0F, 0.0F};
    testCase.beta = {0.0F, 0.0F};
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    testCase.zeroBasedOffsets = {0};
    testCase.zeroBasedColumns.clear();
    // Unequal rows produce 2044 blocks: 128 copy tiles exceed the target
    // AIV count and force each core to reuse UB. The final 12-block tile
    // must not copy stale UB lanes to C.
    for (int32_t row = 0; row < testCase.blockRows; ++row) {
        const int32_t count = row == testCase.blockRows - 1 ? 20 :
            (row % 3 == 0 ? 17 : (row % 3 == 1 ? 35 : 36));
        for (int32_t column = 0; column < count; ++column) {
            testCase.zeroBasedColumns.push_back((column + row * 5) % 37);
        }
        testCase.zeroBasedOffsets.push_back(
            static_cast<int32_t>(testCase.zeroBasedColumns.size()));
    }
    return testCase;
}

BsrComplexCase MakeComplexFourGroupCase(int variant)
{
    BsrComplexCase testCase;
    testCase.name = "BSR_COMPLEX64_B16_FOUR_BLOCK_GROUPS_" + std::to_string(variant);
    testCase.blockRows = 144;
    testCase.blockCols = 64;
    testCase.blockSize = 16;
    testCase.k = 128;
    testCase.blockOrder = variant % 2 == 0 ? ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
    testCase.indexBase = variant == 0 || variant == 3 ?
        ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    const bool identity = variant < 2;
    if (identity) {
        testCase.alpha = {1.0F, 0.0F};
        testCase.beta = {0.0F, 0.0F};
    }
    testCase.zeroBasedOffsets = {0};
    testCase.zeroBasedColumns.clear();
    const std::vector<std::vector<int32_t>> patterns = {
        {}, {63}, {15, 16}, {2, 3, 4}, {7, 8, 9, 10},
        {1, 2, 3, 4, 5}, {2, 2, 3, 4, 5, 12, 13},
        {11, 10, 9, 8, 16, 17, 18, 19},
        {4, 5, 6, 7, 16, 17, 18, 19, 32}};
    // 624 blocks cross core intervals and mix groups with 1/2/3-block
    // row tails, empty rows, gaps, descending runs, and duplicate slots.
    for (int32_t row = 0; row < testCase.blockRows; ++row) {
        auto columns = patterns[static_cast<size_t>(row % 9)];
        if (row % 9 == 4 && (row / 9) % 2 != 0) {
            columns = {60, 61, 62, 63};
        }
        testCase.zeroBasedColumns.insert(testCase.zeroBasedColumns.end(),
            columns.begin(), columns.end());
        testCase.zeroBasedOffsets.push_back(
            static_cast<int32_t>(testCase.zeroBasedColumns.size()));
    }
    return testCase;
}

std::vector<size_t> BuildDescendingComplexReference(
    const BsrComplexCase &testCase, BsrComplexCase &fallback)
{
    std::vector<size_t> originalSlots;
    for (int32_t row = 0; row < testCase.blockRows; ++row) {
        std::vector<size_t> rowSlots;
        const int32_t begin = testCase.zeroBasedOffsets[static_cast<size_t>(row)];
        const int32_t end = testCase.zeroBasedOffsets[static_cast<size_t>(row + 1)];
        for (int32_t slot = begin; slot < end; ++slot) {
            rowSlots.push_back(static_cast<size_t>(slot));
        }
        // Descending order cannot contain an ascending four-column run.
        // Preserve a slot bijection even when equal columns are repeated.
        std::stable_sort(rowSlots.begin(), rowSlots.end(), [&](size_t lhs, size_t rhs) {
            return testCase.zeroBasedColumns[lhs] > testCase.zeroBasedColumns[rhs];
        });
        for (size_t slot : rowSlots) {
            fallback.zeroBasedColumns[originalSlots.size()] = testCase.zeroBasedColumns[slot];
            originalSlots.push_back(slot);
        }
    }
    return originalSlots;
}

BsrComplexCase MakeSparsePackCase(const SparsePackVariant &variant, int ordinal)
{
    BsrComplexCase testCase;
    testCase.name = "BSR_COMPLEX64_SPARSE_Y_CASE_" + std::to_string(ordinal);
    testCase.blockSize = variant.blockSize;
    testCase.blockCols = variant.blockColumns;
    testCase.blockRows = variant.blocks == 64 ? 113 : 4;
    testCase.k = variant.reduction;
    testCase.blockOrder = ordinal % 2 == 0 ? ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
    testCase.indexBase = ordinal % 2 == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    const bool identity = ordinal == 0 || ordinal == 3 || ordinal == 4 || ordinal == 7;
    if (identity) {
        testCase.alpha = {1.0F, 0.0F};
        testCase.beta = {0.0F, 0.0F};
    }
    const int32_t firstCount = std::max(1, variant.blocks / 3);
    const int32_t secondCount = (variant.blocks - firstCount) / 2;
    testCase.zeroBasedOffsets = {0};
    for (int32_t row = 0; row < testCase.blockRows; ++row) {
        const int32_t secondRow = variant.blocks == 64 ? 56 : 2;
        const int32_t count = row == 0 ? firstCount :
            (row == secondRow ? secondCount :
                (row == testCase.blockRows - 1 ?
                    variant.blocks - firstCount - secondCount : 0));
        testCase.zeroBasedOffsets.push_back(testCase.zeroBasedOffsets.back() + count);
    }
    testCase.zeroBasedColumns.clear();
    for (int32_t slot = 0; slot < variant.blocks; ++slot) {
        testCase.zeroBasedColumns.push_back(
            variant.blockColumns - 1 - (slot * 17) % variant.blockColumns);
    }
    if (variant.blocks > 1) {
        // The first and last physical slots share a column across rows;
        // the first two also duplicate it within one row when possible.
        testCase.zeroBasedColumns[1] = testCase.zeroBasedColumns[0];
        testCase.zeroBasedColumns.back() = testCase.zeroBasedColumns[0];
    }
    return testCase;
}

inline std::vector<uint8_t> MakeIdentityBitPayload(
    size_t count, aclDataType type, uint32_t seed)
{
    std::vector<uint8_t> bytes;
    if (type == ACL_COMPLEX64) {
        const auto values = MakeRandomTaskComplexValues(count, seed);
        bytes.resize(count * sizeof(aclsparseComplex));
        std::copy_n(reinterpret_cast<const uint8_t *>(values.data()),
            bytes.size(), bytes.data());
        // Include Inf+0j, signed-zero components and distinct NaN payloads.
        const uint32_t parts[] = {0x7F800000U, 0x00000000U,
            0xFF800000U, 0x80000000U, 0x7FC12345U, 0x3F800000U,
            0x40000000U, 0xFFC54321U, 0x00000001U, 0x80000001U};
        std::copy_n(reinterpret_cast<const uint8_t *>(parts),
            std::min(bytes.size(), sizeof(parts)), bytes.data());
    } else {
        bytes = PackRealValues(MakeRandomTaskValues(count, seed, type), type);
        if (type == ACL_FLOAT) {
            const uint32_t parts[] = {0x7F800000U, 0xFF800000U,
                0x00000000U, 0x80000000U, 0x7FC12345U, 0xFFC54321U,
                0x00000001U, 0x80000001U};
            std::copy_n(reinterpret_cast<const uint8_t *>(parts),
                std::min(bytes.size(), sizeof(parts)), bytes.data());
        } else {
            const uint16_t fp16[] = {0x7C00U, 0xFC00U, 0x0000U, 0x8000U,
                0x7E55U, 0xFE2AU, 0x0001U, 0x8001U};
            const uint16_t bf16[] = {0x7F80U, 0xFF80U, 0x0000U, 0x8000U,
                0x7FC5U, 0xFFD2U, 0x0001U, 0x8001U};
            std::copy_n(reinterpret_cast<const uint8_t *>(type == ACL_BF16 ? bf16 : fp16),
                std::min(bytes.size(), sizeof(fp16)), bytes.data());
        }
    }
    return bytes;
}

inline void CreateDeviceIdentitySparseMat(
    const DeviceIdentityCase &testCase, BsrMatGuard &matC, size_t nnz,
    const SddmmGuardedBuffer &offsets, const SddmmGuardedBuffer &columns,
    const SddmmGuardedBuffer &values)
{
    const int64_t block = testCase.bsr ? 2 : 1;
    if (testCase.bsr) {
        CheckSetupStatus(aclsparseCreateBsr(matC.put(), 4, 4, nnz, block, block,
            offsets.get(), columns.get(), values.get(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, testCase.base, testCase.output, testCase.order),
            "aclsparseCreateBsr");
    } else {
        CheckSetupStatus(aclsparseCreateCsr(matC.put(), 4 * block, 4 * block, nnz,
            offsets.get(), columns.get(), values.get(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, testCase.base, testCase.output), "aclsparseCreateCsr");
    }
}

void RunDeviceIdentityNoOpCase(aclrtStream stream, const DeviceIdentityCase &testCase)
{
    const int64_t block = testCase.bsr ? 2 : 1;
    const int64_t m = 4 * block;
    const int64_t n = 4 * block;
    constexpr int64_t k = 17;
    std::vector<int32_t> offsets = {0, 0, 5, 5, 9};
    std::vector<int32_t> columns = {1, 1, 3, 0, 2, 3, 3, 1, 2};
    for (auto &value : offsets) { value += static_cast<int32_t>(testCase.base); }
    for (auto &value : columns) { value += static_cast<int32_t>(testCase.base); }
    const auto x = MakeIdentityBitPayload(static_cast<size_t>(m * k), testCase.input, 26091231U);
    const auto y = MakeIdentityBitPayload(static_cast<size_t>(k * n), testCase.input, 26091233U);
    const auto initial = MakeIdentityBitPayload(columns.size() * block * block, testCase.output, 26091235U);
    auto dOffsets = SddmmGuardedBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
    auto dColumns = SddmmGuardedBuffer::copyFrom(columns.data(), columns.size() * sizeof(int32_t));
    auto dX = SddmmGuardedBuffer::copyFrom(x.data(), x.size());
    auto dY = SddmmGuardedBuffer::copyFrom(y.data(), y.size());
    auto dValues = SddmmGuardedBuffer::copyFrom(initial.data(), initial.size());
    auto matX = DnMatManager::create(m, k, k, dX.get(), testCase.input, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::create(k, n, n, dY.get(), testCase.input, ACL_SPARSE_ORDER_ROW);
    BsrMatGuard matC;
    CreateDeviceIdentitySparseMat(testCase, matC, columns.size(), dOffsets, dColumns, dValues);
    const aclsparseComplex alpha{testCase.base == ACL_SPARSE_INDEX_BASE_ONE ? -0.0F : 0.0F, -0.0F};
    const aclsparseComplex beta{1.0F, -0.0F};
    const bool complex = testCase.input == ACL_COMPLEX64;
    const size_t scalarBytes = complex ? sizeof(aclsparseComplex) : sizeof(float);
    auto dAlpha = SddmmGuardedBuffer::copyFrom(&alpha, scalarBytes);
    auto dBeta = SddmmGuardedBuffer::copyFrom(&beta, scalarBytes);
    HandleManager handle;
    handle.setStream(stream);
    CheckSetupStatus(aclsparseSetPointerMode(handle.get(), ACL_SPARSE_POINTER_MODE_DEVICE),
        "aclsparseSetPointerMode");
    // Nonempty work, Device-resident scalars and K=17 force the
    // generic Dispatcher to resolve scalars on the device.
    const auto stages = RunGuardedRepeatedStages(handle, dAlpha.get(), dBeta.get(),
        matX, matY, matC.get(), complex ? ACL_COMPLEX64 : ACL_FLOAT, stream,
        dValues, {&dOffsets, &dColumns, &dX, &dY, &dAlpha, &dBeta});
    ASSERT_TRUE(StagesSucceeded(stages));
    EXPECT_EQ(stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    EXPECT_EQ(dValues.snapshot(), initial) << "identity must preserve all C bytes";
}

void SetGuardedBsrPattern(
    int kind, std::vector<int32_t> &offsets, std::vector<int32_t> &columns)
{
    if (kind == 0) {
        offsets = {0, 2, 4, 6};
        columns = {0, 2, 1, 3, 0, 3};
    } else if (kind == 1) {
        offsets = {0, 0, 7, 7};
        columns = {0, 1, 2, 3, 3, 1, 0};
    } else {
        offsets = {0, 1, 1, 2};
        columns = {3, 0};
    }
}

BsrRealCase MakeBlock8IdentityCase(int &ordinal, aclDataType inputType,
    aclDataType outputType, aclsparseOrder_t order, aclsparseIndexBase_t base, int pattern)
{
    BsrRealCase testCase;
    testCase.name = "GUARDED_B8_VECTOR_IDENTITY_" + std::to_string(ordinal++);
    testCase.blockSize = 8;
    testCase.blockRows = 6;
    testCase.blockCols = 8;
    testCase.k = 128;
    testCase.inputType = inputType;
    testCase.outputType = outputType;
    testCase.blockOrder = order;
    testCase.indexBase = base;
    testCase.alpha = 1.0F;
    testCase.beta = 0.0F;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    // More physical blocks than the device's normal AIV grid
    // exercise rowEnd-based X reuse. The first pattern also
    // crosses row boundaries within contiguous owned chunks;
    // the second has one long row surrounded by empty rows.
    testCase.zeroBasedOffsets = pattern == 0 ?
        std::vector<int32_t>{0, 0, 129, 132, 513, 515, 515} :
        std::vector<int32_t>{0, 0, 0, 513, 513, 513, 513};
    testCase.zeroBasedColumns.resize(
        static_cast<size_t>(testCase.zeroBasedOffsets.back()));
    for (size_t slot = 0; slot < testCase.zeroBasedColumns.size(); ++slot) {
        testCase.zeroBasedColumns[slot] =
            static_cast<int32_t>((3U * slot + slot / 11U) % 8U);
    }
    return testCase;
}

void CheckBlock4GenericBroadcast(aclrtStream stream,
    const std::pair<aclDataType, aclDataType> &type, int layout, int base,
    int scalars, int &ordinal)
{
    BsrRealCase testCase;
    testCase.name = "BSR_REAL_B4_K32_GENERIC_CASE_" + std::to_string(ordinal++);
    testCase.blockRows = 6;
    testCase.blockCols = 43;
    testCase.blockSize = 4;
    testCase.k = 32;
    testCase.inputType = type.first;
    testCase.outputType = type.second;
    testCase.blockOrder = layout == 0 ? ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
    testCase.indexBase = base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    testCase.deviceScalars = (layout + base + scalars) % 2 != 0;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    testCase.zeroBasedOffsets = {0, 0, 1, 2, 1145, 1145, 1201};
    testCase.zeroBasedColumns.clear();
    for (int slot = 0; slot < 1201; ++slot) {
        testCase.zeroBasedColumns.push_back((slot * 17) % 43);
    }
    testCase.initialValuesOverride = MakeRandomTaskValues(
        1201U * 16U, 26091203U, testCase.outputType);
    if (scalars == 0) {
        testCase.alpha = 1.0F;
        testCase.beta = 0.0F;
    } else if (scalars == 2) {
        testCase.alpha = 0.0F;
        testCase.beta = -0.5F;
        testCase.specialDenseValues = true;
    }
    SCOPED_TRACE(testCase.name);
    const auto specialized = RunBsrRealCase(stream, testCase);
    ASSERT_TRUE(StagesSucceeded(specialized.stages));
    ASSERT_TRUE(VerifyRealRun(specialized, testCase.outputType, testCase.name));
    // Two output batches with broadcast X/Y disable the
    // single-batch fast path. Copy C explicitly: the random
    // generator shuffles by total count, so equal seeds alone
    // would not preserve the first batch for general beta.
    testCase.outputBatchCount = 2;
    const auto firstBatch = testCase.initialValuesOverride;
    testCase.initialValuesOverride.insert(testCase.initialValuesOverride.end(),
        firstBatch.begin(), firstBatch.end());
    testCase.name += "_GENERIC_BATCH";
    const auto generic = RunBsrRealCase(stream, testCase);
    ASSERT_TRUE(StagesSucceeded(generic.stages));
    ASSERT_TRUE(VerifyRealRun(generic, testCase.outputType, testCase.name));
    ASSERT_EQ(generic.actual.size(), specialized.actual.size() * 2U);
    ASSERT_EQ(std::memcmp(specialized.actual.data(), generic.actual.data(),
        specialized.actual.size() * sizeof(float)), 0);
}

void CheckBsr4TailCase(aclrtStream stream,
    const std::pair<aclDataType, aclDataType> &type, int32_t blockNnz,
    int ordinal, bool deviceScalars, int scalarMode)
{
    BsrRealCase testCase;
    testCase.name = "BSR_REAL_B4_K32_TAIL_" + std::to_string(ordinal) +
        "_BLOCKS_" + std::to_string(blockNnz);
    testCase.blockRows = 6;
    testCase.blockCols = 43;
    testCase.blockSize = 4;
    testCase.k = 32;
    testCase.inputType = type.first;
    testCase.outputType = type.second;
    testCase.indexBase = ordinal % 2 == 0 ?
        ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
    testCase.blockOrder = (ordinal / 2) % 2 == 0 ?
        ACL_SPARSE_ORDER_ROW : ACL_SPARSE_ORDER_COL;
    testCase.deviceScalars = deviceScalars;
    testCase.alpha = -0.625F;
    testCase.beta = 0.375F;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    // Leading, interior and trailing empty rows exercise sparse row lookup.
    // Repeated columns in the long row still address distinct output slots.
    testCase.zeroBasedOffsets = {0, 0, 1, 1, blockNnz, blockNnz, blockNnz};
    testCase.zeroBasedColumns.clear();
    for (int32_t slot = 0; slot < blockNnz; ++slot) {
        testCase.zeroBasedColumns.push_back((slot % 43) * 17 % 43);
    }
    if (scalarMode == 0) {
        testCase.alpha = 1.0F;
        testCase.beta = 0.0F;
    } else if (scalarMode == 2) {
        testCase.alpha = 0.0F;
        testCase.beta = -0.5F;
        testCase.specialDenseValues = true;
    }
    SCOPED_TRACE(testCase.name);
    std::cout << testCase.name << " input=" << type.first << " output=" << type.second
              << " base=" << testCase.indexBase << " order=" << testCase.blockOrder
              << " deviceScalars=" << deviceScalars << " alpha=" << testCase.alpha
              << " beta=" << testCase.beta << std::endl;
    const auto result = RunBsrRealCase(stream, testCase);
    ASSERT_TRUE(StagesSucceeded(result.stages));
    ASSERT_EQ(result.actual.size(), static_cast<size_t>(blockNnz) * 16U);
    EXPECT_EQ(result.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    // RunBsrRealCase supplies the independent CPU golden. guardedRepeat checks
    // all input/output/workspace canaries and repeat identity after restoring C.
    ASSERT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
}

SddmmGuardedBuffer SddmmGuardedBuffer::copyFrom(
    const void *source, size_t bytes, size_t extraOffset)
{
    SddmmGuardedBuffer result;
    result.bytes_ = bytes;
    result.prefix_ = 64U + extraOffset;
    if (bytes == 0) {
        return result;
    }
    result.initial_.resize(bytes, 0);
    if (source != nullptr) {
        std::copy_n(static_cast<const uint8_t *>(source), bytes,
                    result.initial_.begin());
    }
    std::vector<uint8_t> storage(result.prefix_ + bytes + 64U, 0xA7U);
    std::copy(result.initial_.begin(), result.initial_.end(),
              storage.begin() + result.prefix_);
    result.storage_ = DeviceBuffer::copyFrom(storage.data(), storage.size());
    return result;
}

void *SddmmGuardedBuffer::get() const
{
    return bytes_ == 0 ? nullptr :
        static_cast<uint8_t *>(storage_.raw()) + prefix_;
}

void SddmmGuardedBuffer::copyToHost(void *destination, size_t bytes) const
{
    if (bytes > bytes_ || (bytes != 0 && aclrtMemcpy(
            destination, bytes, get(), bytes,
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("guarded buffer D2H failed");
    }
}

std::vector<uint8_t> SddmmGuardedBuffer::snapshot() const
{
    std::vector<uint8_t> result(bytes_);
    copyToHost(result.data(), result.size());
    return result;
}

bool SddmmGuardedBuffer::unchanged() const
{ return snapshot() == initial_; }

bool SddmmGuardedBuffer::guardsIntact() const
{
    if (bytes_ == 0) {
        return get() == nullptr;
    }
    std::vector<uint8_t> before(prefix_);
    std::vector<uint8_t> after(64U);
    if (aclrtMemcpy(before.data(), before.size(), storage_.raw(),
            before.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(after.data(), after.size(),
            static_cast<uint8_t *>(get()) + bytes_, after.size(),
            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("guarded buffer canary D2H failed");
    }
    const auto isCanary = [](uint8_t byte) { return byte == 0xA7U; };
    return std::all_of(before.begin(), before.end(), isCanary) &&
           std::all_of(after.begin(), after.end(), isCanary);
}

void SddmmGuardedBuffer::restore() const
{
    if (bytes_ != 0 && aclrtMemcpy(get(), bytes_, initial_.data(), bytes_,
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("guarded buffer restore failed");
    }
}

template <typename Scalar>
void BsrTestResources::SetScalars(aclrtStream stream, bool deviceScalars,
                const Scalar &hostAlpha, const Scalar &hostBeta)
{
    handle.setStream(stream);
    CheckSetupStatus(aclsparseSetPointerMode(handle.get(), deviceScalars ?
        ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        "aclsparseSetPointerMode");
    alpha = &hostAlpha;
    beta = &hostBeta;
    if (deviceScalars) {
        dAlpha = DeviceBuffer::copyFrom(&hostAlpha, sizeof(Scalar));
        dBeta = DeviceBuffer::copyFrom(&hostBeta, sizeof(Scalar));
        alpha = dAlpha.get();
        beta = dBeta.get();
    }
}

BsrWorkspaceFixture::BsrWorkspaceFixture(aclrtStream callStream) : stream(callStream)
{}

template <typename Case, typename Value>
void BsrWorkspaceFixture::Initialize(const Case &testCase, aclDataType valueType,
    const std::vector<Value> &x, const std::vector<Value> &y,
    const std::vector<Value> &initial, size_t denseOffset)
{
    auto offsets = testCase.zeroBasedOffsets;
    auto columns = testCase.zeroBasedColumns;
    for (auto &value : offsets) {
        value += static_cast<int32_t>(testCase.indexBase);
    }
    for (auto &value : columns) {
        value += static_cast<int32_t>(testCase.indexBase);
    }
    dOffsets = SddmmGuardedBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
    dColumns = SddmmGuardedBuffer::copyFrom(columns.data(), columns.size() * sizeof(int32_t));
    dX = SddmmGuardedBuffer::copyFrom(x.data(), x.size() * sizeof(Value), denseOffset);
    dY = SddmmGuardedBuffer::copyFrom(y.data(), y.size() * sizeof(Value), denseOffset);
    dValues = SddmmGuardedBuffer::copyFrom(initial.data(), initial.size() * sizeof(Value));
    const int64_t m = testCase.blockRows * testCase.blockSize;
    const int64_t n = testCase.blockCols * testCase.blockSize;
    matX = DnMatManager::create(m, testCase.k, testCase.k, dX.get(), valueType, ACL_SPARSE_ORDER_ROW);
    matY = DnMatManager::create(testCase.k, n, n, dY.get(), valueType, ACL_SPARSE_ORDER_ROW);
    CreateBsrTestDescriptor(matC, testCase, valueType, dOffsets.get(), dColumns.get(), dValues.get());
    computeType = valueType;
    SetScalars(stream, testCase.deviceScalars, testCase.alpha, testCase.beta);
}

bool BsrWorkspaceFixture::Query()
{
    const auto before = dValues.snapshot();
    const bool success = QueryTestWorkspace(stages, handle, alpha, beta,
        matX, matY, matC.get(), computeType);
    EXPECT_EQ(before, dValues.snapshot()) << "BufferSize must preserve C";
    CheckGuardedBuffers(dValues, guardedWorkspace, {&dOffsets, &dColumns, &dX, &dY});
    return success;
}

bool BsrWorkspaceFixture::Run(bool preprocess)
{
    dValues.restore();
    const std::vector<const SddmmGuardedBuffer *> inputs = {&dOffsets, &dColumns, &dX, &dY};
    if (preprocess) {
        stages.preprocessRet = aclsparseSDDMMPreprocess(
            handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            alpha, matX.cget(), matY.cget(), beta, matC.get(), computeType,
            ACL_SPARSE_SDDMM_ALG_DEFAULT, guardedWorkspace.get());
        if (stages.preprocessRet != ACL_SPARSE_STATUS_SUCCESS ||
            aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return false;
        }
        EXPECT_TRUE(dValues.unchanged()) << "Preprocess must preserve C";
        CheckGuardedBuffers(dValues, guardedWorkspace, inputs);
    }
    RepeatGuardedExecute(stages, handle, alpha, beta, matX, matY, matC.get(),
        computeType, stream, dValues, guardedWorkspace, inputs);
    return stages.executeRet == ACL_SPARSE_STATUS_SUCCESS && stages.synchronizeRet == ACL_SUCCESS;
}

template void BsrTestResources::SetScalars<float>(
    aclrtStream, bool, const float &, const float &);
template void BsrTestResources::SetScalars<aclsparseComplex>(
    aclrtStream, bool, const aclsparseComplex &, const aclsparseComplex &);
template void BsrWorkspaceFixture::Initialize<BsrRealCase, float>(
    const BsrRealCase &, aclDataType, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &, size_t);
template void BsrWorkspaceFixture::Initialize<BsrComplexCase, aclsparseComplex>(
    const BsrComplexCase &, aclDataType, const std::vector<aclsparseComplex> &,
    const std::vector<aclsparseComplex> &, const std::vector<aclsparseComplex> &, size_t);
} // namespace sddmm_extended
} // namespace sparse_test
