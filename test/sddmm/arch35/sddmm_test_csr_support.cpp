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
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace sparse_test {
namespace sddmm_extended {

inline std::vector<float> CsrRealGolden(
    int64_t m, int64_t n, int64_t k,
    const std::vector<int32_t> &rowOffsets,
    const std::vector<int32_t> &columns,
    const std::vector<float> &x, const std::vector<float> &y,
    const std::vector<float> &cInitial, float alpha, float beta)
{
    (void)n;
    std::vector<float> result(columns.size());
    for (int64_t row = 0; row < m; ++row) {
        const int64_t begin = rowOffsets[static_cast<size_t>(row)];
        const int64_t end = rowOffsets[static_cast<size_t>(row + 1)];
        for (int64_t position = begin; position < end; ++position) {
            const int64_t column = columns[static_cast<size_t>(position)];
            double dot = 0.0;
            for (int64_t reduction = 0; reduction < k; ++reduction) {
                dot = std::fma(
                    static_cast<double>(x[static_cast<size_t>(row * k + reduction)]),
                    static_cast<double>(y[static_cast<size_t>(reduction * n + column)]), dot);
            }
            result[static_cast<size_t>(position)] = static_cast<float>(
                static_cast<double>(alpha) * dot + static_cast<double>(beta) *
                    cInitial[static_cast<size_t>(position)]);
        }
    }
    return result;
}

inline void ValidateCsrTestShape(int64_t m, int64_t n, int64_t k, int64_t rowStride)
{
    if (m < 0 || n <= 0 || k <= 0 || rowStride < 0 ||
        m > std::numeric_limits<int32_t>::max() / 64 ||
        n > std::numeric_limits<int32_t>::max() ||
        k > std::numeric_limits<int32_t>::max() ||
        (m > 0 && rowStride > (std::numeric_limits<int64_t>::max() - n) / m)) {
        throw std::invalid_argument("invalid CSR test dimensions or row stride");
    }
}

inline int64_t CsrColumnRemainder(int64_t value, int64_t columnCount)
{
    if (columnCount > 0) {
        return value % columnCount;
    }
    throw std::invalid_argument("invalid CSR test dimensions or row stride");
}

inline void MakeCyclicCsrPattern(int64_t m, int64_t n, int64_t rowStride,
    int64_t firstColumn, int32_t base, bool perturbOneColumn,
    std::vector<int32_t> &offsets, std::vector<int32_t> &columns)
{
    if (n <= 0) {
        throw std::invalid_argument("invalid CSR test dimensions or row stride");
    }
    ValidateCsrTestShape(m, n, 128, rowStride);
    if (firstColumn < 0 || firstColumn > std::numeric_limits<int32_t>::max() ||
        (m > 0 && rowStride > (std::numeric_limits<int64_t>::max() - firstColumn) / m)) {
        throw std::invalid_argument("invalid CSR first column");
    }
    constexpr int32_t nnzPerRow = 64;
    offsets.resize(static_cast<size_t>(m + 1));
    columns.resize(static_cast<size_t>(m * nnzPerRow));
    for (int64_t row = 0; row < m; ++row) {
        offsets[static_cast<size_t>(row)] = static_cast<int32_t>(row * nnzPerRow) + base;
        const int64_t first = CsrColumnRemainder(firstColumn + row * rowStride, n);
        auto begin = columns.begin() + row * nnzPerRow;
        for (int32_t slot = 0; slot < nnzPerRow; ++slot) {
            begin[slot] = static_cast<int32_t>(CsrColumnRemainder(first + slot, n));
        }
        if (perturbOneColumn && row == 7) {
            // Replace one column outside the original cyclic interval.
            begin[31] = static_cast<int32_t>(CsrColumnRemainder(first + 96, n));
        }
        std::sort(begin, begin + nnzPerRow);
        for (int32_t slot = 0; slot < nnzPerRow; ++slot) {
            begin[slot] += base;
        }
    }
    offsets.back() = static_cast<int32_t>(columns.size()) + base;
}

inline void FillCsrFp16ColumnGolden(RealRunResult &result, int64_t m, int64_t n,
    int32_t base, const std::vector<int32_t> &offsets,
    const std::vector<int32_t> &columns, const std::vector<float> &x,
    const std::vector<float> &yLogical)
{
    constexpr int64_t k = 128;
    result.golden.resize(columns.size());
    for (int64_t row = 0; row < m; ++row) {
        const int64_t begin = offsets[static_cast<size_t>(row)] - base;
        const int64_t end = offsets[static_cast<size_t>(row + 1)] - base;
        for (int64_t position = begin; position < end; ++position) {
            const int64_t column = columns[static_cast<size_t>(position)] - base;
            float dot = 0.0F;
            for (int64_t reduction = 0; reduction < k; ++reduction) {
                dot = std::fma(
                    x[static_cast<size_t>(row * k + reduction)],
                    yLogical[static_cast<size_t>(reduction * n + column)], dot);
            }
            result.golden[static_cast<size_t>(position)] =
                QuantizeReal(dot, ACL_FLOAT16);
        }
    }
}

inline std::vector<float> PackColumnMajor(const std::vector<float> &rowMajor,
    int64_t rows, int64_t columns)
{
    std::vector<float> result(rowMajor.size());
    for (int64_t column = 0; column < columns; ++column) {
        for (int64_t row = 0; row < rows; ++row) {
            result[static_cast<size_t>(column * rows + row)] =
                rowMajor[static_cast<size_t>(row * columns + column)];
        }
    }
    return result;
}

struct CsrColumnTestDevices {
    DeviceBuffer offsets;
    DeviceBuffer columns;
    DeviceBuffer x;
    DeviceBuffer y;
    DeviceBuffer values;
    DnMatManager matX;
    DnMatManager matY;
    SpMatManager matC;

    CsrColumnTestDevices(int64_t m, int64_t n, int64_t k, aclsparseIndexBase_t base,
        const std::vector<int32_t> &hostOffsets, const std::vector<int32_t> &hostColumns,
        const std::vector<uint8_t> &hostX, const std::vector<uint8_t> &hostY,
        const std::vector<uint8_t> &hostValues)
    {
        offsets = DeviceBuffer::copyFrom(hostOffsets.data(), hostOffsets.size() * sizeof(int32_t));
        columns = DeviceBuffer::copyFrom(hostColumns.data(), hostColumns.size() * sizeof(int32_t));
        x = DeviceBuffer::copyFrom(hostX.data(), hostX.size());
        y = DeviceBuffer::copyFrom(hostY.data(), hostY.size());
        values = DeviceBuffer::copyFrom(hostValues.data(), hostValues.size());
        matX = DnMatManager::createConst(m, k, k, x.get(), ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
        matY = DnMatManager::createConst(k, n, k, y.get(), ACL_FLOAT16, ACL_SPARSE_ORDER_COL);
        matC = SpMatManager::createCsr(m, n, static_cast<int64_t>(hostColumns.size()),
            offsets.get(), columns.get(), values.get(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, base, ACL_FLOAT16);
    }
};

RealRunResult RunCsrFp16ColumnMajorCase(
    aclrtStream stream, aclsparseIndexBase_t indexBase,
    int64_t m, int64_t n,
    int64_t rowStride)
{
    constexpr int64_t k = 128;
    constexpr uint32_t seed = 20260912;
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const int32_t base = static_cast<int32_t>(indexBase);

    ValidateCsrTestShape(m, n, k, rowStride);
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
    MakeCyclicCsrPattern(m, n, rowStride, seed, base, false, offsets, columns);

    const auto x = QuantizeRealVector(
        MakeRealValues(1, m * k, 0.0625F), ACL_FLOAT16);
    const auto yLogical = QuantizeRealVector(
        MakeRealValues(1, k * n, -0.03125F), ACL_FLOAT16);
    const auto yColumnMajor = PackColumnMajor(yLogical, k, n);
    const std::vector<float> cInitial(columns.size(), 0.0F);
    auto xBytes = PackRealValues(x, ACL_FLOAT16);
    auto yBytes = PackRealValues(yColumnMajor, ACL_FLOAT16);
    auto cBytes = PackRealValues(cInitial, ACL_FLOAT16);

    CsrColumnTestDevices devices(m, n, k, indexBase, offsets, columns,
        xBytes, yBytes, cBytes);

    HandleManager handle;
    handle.setStream(stream);
    DeviceBuffer workspace;
    RealRunResult result;
    result.stages = RunThreeStages(
        handle, &alpha, &beta, devices.matX, devices.matY, devices.matC.get(), ACL_FLOAT,
        stream, workspace);
    result.patternFlags = devices.matC.get()->sddmmPrepared.patternFlags;
    result.preparedLogicalFirstColumn =
        devices.matC.get()->sddmmPrepared.patternLogicalFirstColumn;
    if (result.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS ||
        result.stages.synchronizeRet != ACL_SUCCESS) {
        return result;
    }

    std::vector<uint8_t> outputBytes(cBytes.size());
    devices.values.copyToHost(outputBytes.data(), outputBytes.size());
    result.actual = UnpackRealValues(
        outputBytes, columns.size(), ACL_FLOAT16);
    FillCsrFp16ColumnGolden(result, m, n, base, offsets, columns, x, yLogical);
    return result;
}

inline void FillCsrCubeGolden(CsrCubeRealRunResult &result,
    const CsrCubeRealCase &testCase, const std::vector<int32_t> &offsets,
    const std::vector<int32_t> &columns, const std::vector<float> &x,
    const std::vector<float> &y, const std::vector<float> &cInitial)
{
    const int64_t m = testCase.m;
    const int64_t n = testCase.n;
    constexpr int64_t k = 128;
    const int32_t base = static_cast<int32_t>(testCase.indexBase);
    const aclDataType outputType = testCase.fp32Output ? ACL_FLOAT : testCase.dataType;
    result.run.golden.resize(columns.size());
    for (int64_t row = 0; row < m; ++row) {
        const int64_t begin = offsets[static_cast<size_t>(row)] - base;
        const int64_t end = offsets[static_cast<size_t>(row + 1)] - base;
        for (int64_t position = begin; position < end; ++position) {
            const int64_t column =
                columns[static_cast<size_t>(position)] - base;
            const float value = RealDotGolden(
                x.data() + row * k, y.data() + column, k, n,
                testCase.dataType, testCase.alpha, testCase.beta,
                cInitial[static_cast<size_t>(position)]);
            result.run.golden[static_cast<size_t>(position)] =
                QuantizeReal(value, outputType);
        }
    }
}

class CsrCubeRealCaseRunner {
public:
    CsrCubeRealCaseRunner(aclrtStream callStream, const CsrCubeRealCase &params)
        : testCase(params), stream(callStream) {}

    CsrCubeRealRunResult Run()
    {
        PrepareInputs();
        CreateDescriptors();
        PrepareExpectedWorkspace();
        return Execute();
    }

private:
    const CsrCubeRealCase &testCase;
    aclrtStream stream;
    int64_t m{};
    int64_t n{};
    int64_t k{};
    int32_t base{};
    aclDataType outputType{};
    std::vector<int32_t> offsets{};
    std::vector<int32_t> columns{};
    std::vector<float> x{};
    std::vector<float> y{};
    std::vector<float> cInitial{};
    std::vector<uint8_t> xBytes{};
    std::vector<uint8_t> yBytes{};
    std::vector<uint8_t> cBytes{};
    DeviceBuffer dOffsets{};
    DeviceBuffer dColumns{};
    DeviceBuffer dX{};
    DeviceBuffer dY{};
    DeviceBuffer dValues{};
    DeviceBuffer workspace{};
    DnMatManager matX{};
    DnMatManager matY{};
    SpMatManager matC{};
    HandleManager handle{};
    CsrCubeRealRunResult result{};

    void PrepareInputs()
    {
        m = testCase.m;
        n = testCase.n;
        k = 128;
        base = static_cast<int32_t>(testCase.indexBase);
        outputType = testCase.fp32Output ? ACL_FLOAT : testCase.dataType;

        ValidateCsrTestShape(m, n, k, testCase.rowStride);
        MakeCyclicCsrPattern(m, n, testCase.rowStride, testCase.logicalFirstColumn,
            base, testCase.perturbOneColumn, offsets, columns);

        x = QuantizeRealVector(
            MakeRealValues(1, m * k, 0.0625F), testCase.dataType);
        y = QuantizeRealVector(
            MakeRealValues(1, k * n, -0.03125F), testCase.dataType);
        if (testCase.specialDenseValues) {
            std::fill(x.begin(), x.end(),
                std::numeric_limits<float>::quiet_NaN());
            std::fill(y.begin(), y.end(),
                std::numeric_limits<float>::infinity());
        }
        cInitial = QuantizeRealVector(
            MakeRealValues(1, static_cast<int64_t>(columns.size()), 0.015625F),
            outputType);
        xBytes = PackRealValues(x, testCase.dataType);
        yBytes = PackRealValues(y, testCase.dataType);
        cBytes = PackRealValues(cInitial, outputType);
    }

    void CreateDescriptors()
    {
        dOffsets = DeviceBuffer::copyFrom(
            offsets.data(), offsets.size() * sizeof(int32_t));
        dColumns = DeviceBuffer::copyFrom(
            columns.data(), columns.size() * sizeof(int32_t));
        dX = DeviceBuffer::copyFrom(xBytes.data(), xBytes.size());
        dY = DeviceBuffer::copyFrom(yBytes.data(), yBytes.size());
        dValues = DeviceBuffer::copyFrom(cBytes.data(), cBytes.size());
        matX = DnMatManager::createConst(
            m, k, k, dX.get(), testCase.dataType, ACL_SPARSE_ORDER_ROW);
        matY = DnMatManager::createConst(
            k, n, n, dY.get(), testCase.dataType, ACL_SPARSE_ORDER_ROW);
        matC = SpMatManager::createCsr(
            m, n, static_cast<int64_t>(columns.size()), dOffsets.get(),
            dColumns.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
            ACL_SPARSE_INDEX_32I, testCase.indexBase, outputType);
    }

    void PrepareExpectedWorkspace()
    {
        SetHostTestStream(handle, stream);
        const bool canPackY = testCase.dataType == ACL_FLOAT ||
            testCase.dataType == ACL_FLOAT16;
        result.expectedPackedYBytes = canPackY ?
            static_cast<size_t>(k * n) * RealTypeBytes(testCase.dataType) : 0U;
        result.expectedCubeScratchBytes = static_cast<size_t>(
            sddmm_arch35::SddmmCsrCubeDenseWorkspaceBytes(m));
        result.expectedStaticWorkspace =
            static_cast<size_t>(kSddmmWorkspaceBytes) +
            std::max(result.expectedPackedYBytes,
                     result.expectedCubeScratchBytes);
    }

    CsrCubeRealRunResult Execute()
    {
        result.run.stages = RunThreeStages(
            handle, &testCase.alpha, &testCase.beta, matX, matY, matC.get(),
            ACL_FLOAT, stream, workspace);
        const auto *rawDescr = matC.get();
        result.patternFlags = rawDescr->sddmmPrepared.patternFlags;
        result.preparedLogicalFirstColumn =
            rawDescr->sddmmPrepared.patternLogicalFirstColumn;
        if (result.run.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS ||
            result.run.stages.synchronizeRet != ACL_SUCCESS) {
            return result;
        }

        std::vector<uint8_t> outputBytes(cBytes.size());
        dValues.copyToHost(outputBytes.data(), outputBytes.size());
        result.run.actual = UnpackRealValues(
            outputBytes, columns.size(), outputType);
        FillCsrCubeGolden(result, testCase, offsets, columns, x, y, cInitial);
        return result;
    }
};

CsrCubeRealRunResult RunCsrCubeRealCase(aclrtStream stream, const CsrCubeRealCase &testCase)
{
    CsrCubeRealCaseRunner runner(stream, testCase);
    return runner.Run();
}

struct SmallCsrStateFixture {
    static constexpr int64_t m = 2;
    static constexpr int64_t n = 3;
    static constexpr int64_t k = 2;
    float alpha = 0.75F;
    float beta = -0.25F;
    std::vector<int32_t> offsets{0, 1, 3};
    std::vector<int32_t> columns{0, 1, 2};
    std::vector<float> x{0.25F, -0.5F, 0.75F, 0.125F};
    std::vector<float> y{0.5F, -0.25F, 0.75F, -0.125F, 0.625F, -0.375F};
    std::vector<float> values{0.125F, -0.25F, 0.5F};
    DeviceBuffer dOffsets = DeviceBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
    DeviceBuffer dColumns = DeviceBuffer::copyFrom(columns.data(), columns.size() * sizeof(int32_t));
    DeviceBuffer dValues = DeviceBuffer::copyFrom(values.data(), values.size() * sizeof(float));
    DeviceBuffer dX = DeviceBuffer::copyFrom(x.data(), x.size() * sizeof(float));
    DeviceBuffer dY = DeviceBuffer::copyFrom(y.data(), y.size() * sizeof(float));
    DnMatManager matX = DnMatManager::createConst(m, k, k, dX.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    DnMatManager matY = DnMatManager::createConst(k, n, n, dY.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    SpMatManager matC = SpMatManager::createCsr(
        m, n, static_cast<int64_t>(columns.size()), dOffsets.get(), dColumns.get(), dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    HandleManager handle;

    explicit SmallCsrStateFixture(aclrtStream stream)
    {
        handle.setStream(stream);
    }
};

inline bool PrepareSmallCsrState(SmallCsrStateFixture &fixture,
    SddmmStageResult &stages, DeviceBuffer &workspace)
{
    if (!QueryTestWorkspace(stages, fixture.handle, &fixture.alpha, &fixture.beta,
            fixture.matX, fixture.matY, fixture.matC.get(), ACL_FLOAT)) {
        return false;
    }
    if (stages.workspaceSize > 0) {
        workspace = DeviceBuffer::alloc(stages.workspaceSize);
        stages.workspacePointerAligned = reinterpret_cast<uintptr_t>(workspace.get()) %
            kSddmmWorkspaceAlignment == 0;
    }
    stages.preprocessRet = aclsparseSDDMMPreprocess(
        fixture.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &fixture.alpha, fixture.matX.cget(), fixture.matY.cget(), &fixture.beta,
        fixture.matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace.get());
    return stages.preprocessRet == ACL_SPARSE_STATUS_SUCCESS;
}

PreparedInvalidationResult RunPreparedInvalidationCase(
    aclrtStream stream)
{
    SmallCsrStateFixture fixture(stream);
    const std::vector<int32_t> offsetsTwo = {0, 2, 3};
    const std::vector<int32_t> columnsTwo = {1, 2, 0};
    const std::vector<float> cTwo = {-0.375F, 0.25F, 0.0625F};
    auto dOffsetsTwo = DeviceBuffer::copyFrom(offsetsTwo.data(), offsetsTwo.size() * sizeof(int32_t));
    auto dColumnsTwo = DeviceBuffer::copyFrom(columnsTwo.data(), columnsTwo.size() * sizeof(int32_t));
    auto dValuesTwo = DeviceBuffer::copyFrom(cTwo.data(), cTwo.size() * sizeof(float));
    PreparedInvalidationResult result;
    DeviceBuffer workspace;
    if (!PrepareSmallCsrState(fixture, result.stages, workspace)) {
        return result;
    }

    auto *rawDescr = fixture.matC.get();
    result.preparedBefore = rawDescr->sddmmPrepared.prepared;
    result.generationBefore = rawDescr->structureGeneration;
    result.epochBefore = rawDescr->sddmmPrepared.preprocessEpoch;
    result.setterRet = aclsparseCsrSetPointers(
        fixture.matC.get(), dOffsetsTwo.get(), dColumnsTwo.get(), dValuesTwo.get());
    result.preparedAfterSetter = rawDescr->sddmmPrepared.prepared;
    result.workspaceClearedAfterSetter =
        rawDescr->sddmmPrepared.workspace == nullptr;
    result.generationAfterSetter = rawDescr->structureGeneration;
    if (result.setterRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }

    result.stages.executeRet = aclsparseSDDMM(
        fixture.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &fixture.alpha, fixture.matX.cget(), fixture.matY.cget(),
        &fixture.beta, fixture.matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    if (result.stages.executeRet != ACL_SPARSE_STATUS_SUCCESS) {
        return result;
    }
    result.stages.synchronizeRet = aclrtSynchronizeStream(stream);
    if (result.stages.synchronizeRet != ACL_SUCCESS) {
        return result;
    }
    result.preparedAfterExecute = rawDescr->sddmmPrepared.prepared;
    result.epochAfter = rawDescr->sddmmPrepared.preprocessEpoch;
    result.rebuiltForNewOffsets =
        rawDescr->sddmmPrepared.key.offsets == dOffsetsTwo.get();
    result.actual.resize(cTwo.size());
    dValuesTwo.copyToHost(result.actual.data(), result.actual.size() * sizeof(float));
    result.golden = CsrRealGolden(
        fixture.m, fixture.n, fixture.k, offsetsTwo, columnsTwo, fixture.x, fixture.y, cTwo, fixture.alpha, fixture.beta);
    return result;
}

ActiveBufferIsolationResult RunActiveBufferIsolationCase(
    aclrtStream stream)
{
    SmallCsrStateFixture fixture(stream);

    ActiveBufferIsolationResult result;
    size_t workspaceSize = 0;
    result.bufferSizeRet = aclsparseSDDMMBufferSize(
        fixture.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &fixture.alpha, fixture.matX.cget(), fixture.matY.cget(),
        &fixture.beta, fixture.matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        &workspaceSize);
    if (result.bufferSizeRet != ACL_SPARSE_STATUS_SUCCESS ||
        workspaceSize == 0) {
        return result;
    }
    DeviceBuffer workspace = DeviceBuffer::alloc(workspaceSize);
    DeviceBuffer distinctWorkspace = DeviceBuffer::alloc(workspaceSize);
    auto *rawDescr = fixture.matC.get();

    rawDescr->activeBuffer = distinctWorkspace.get();
    rawDescr->activeSpmvBuffer = distinctWorkspace.get();
    result.distinctWorkspaceRet = aclsparseSDDMMPreprocess(
        fixture.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &fixture.alpha, fixture.matX.cget(), fixture.matY.cget(),
        &fixture.beta, fixture.matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    result.distinctOwnerPreserved =
        rawDescr->activeBuffer == distinctWorkspace.get();
    result.distinctSpmvOwnerPreserved =
        rawDescr->activeSpmvBuffer == distinctWorkspace.get();

    rawDescr->activeBuffer = workspace.get();
    rawDescr->activeSpmvBuffer = workspace.get();
    result.aliasedWorkspaceRet = aclsparseSDDMMPreprocess(
        fixture.handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, &fixture.alpha, fixture.matX.cget(), fixture.matY.cget(),
        &fixture.beta, fixture.matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT,
        workspace.get());
    result.aliasedOwnerCleared = rawDescr->activeBuffer == nullptr;
    result.aliasedSpmvOwnerCleared = rawDescr->activeSpmvBuffer == nullptr;
    result.sddmmPrepared = rawDescr->sddmmPrepared.prepared;
    return result;
}

void CheckCsrCubeNumerics(const CsrCubeRealRunResult &result,
    const CsrCubeRealCase &testCase, bool scratchDominates)
{
    if (scratchDominates) {
        ASSERT_GT(result.expectedCubeScratchBytes, result.expectedPackedYBytes);
        EXPECT_EQ(result.run.stages.workspaceSize,
            static_cast<size_t>(kSddmmWorkspaceBytes) + result.expectedCubeScratchBytes);
    }
    const aclDataType outputType = testCase.fp32Output ? ACL_FLOAT : testCase.dataType;
    EXPECT_TRUE(VerifyRealRun(result.run, outputType, testCase.name));
}

static std::vector<double> Fp16ToDoubles(const std::vector<uint16_t>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = static_cast<double>(Fp16BitsToFp32(v[i]));
    }
    return out;
}

static std::vector<double> QuantizeDoublesViaFp16(const std::vector<double>& v) {
    return Fp16ToDoubles(DoublesToFp16(v));
}

template <typename T>
static std::vector<T> RepackRowToCol(const std::vector<T>& rowBuf,
                                      int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        return rowBuf;
    }
    std::vector<T> colBuf(rowBuf.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            colBuf[static_cast<size_t>(j) * static_cast<size_t>(rows) +
                   static_cast<size_t>(i)] =
                rowBuf[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                       static_cast<size_t>(j)];
        }
    }
    return colBuf;
}

SddmmInputs GenerateSddmmInputs(const SddmmTestParam& p) {
    SddmmInputs in;
    in.csrC = MakeSddmmSparsity(p.m, p.n, p.sparsity_ratio,
                                p.value_lo, p.value_hi, p.random_seed);
    in.nnz = in.csrC.nnz;

    std::mt19937 rngX(p.random_seed + 1);
    std::mt19937 rngY(p.random_seed + 2);
    std::uniform_real_distribution<double> dist(p.value_lo, p.value_hi);

    in.Xf64.resize(static_cast<size_t>(p.m) * static_cast<size_t>(p.k));
    in.Yf64.resize(static_cast<size_t>(p.n) * static_cast<size_t>(p.k));
    for (size_t i = 0; i < in.Xf64.size(); i++) {
        in.Xf64[i] = dist(rngX);
    }
    for (size_t i = 0; i < in.Yf64.size(); i++) {
        in.Yf64[i] = dist(rngY);
    }
    return in;
}

std::vector<double> ComputeGoldenExpect(const SddmmTestParam& p,
                                                const SddmmCsr& csrC,
                                                const std::vector<double>& Xf64,
                                                const std::vector<double>& Yf64,
                                                aclDataType dtype,
                                                aclsparseOperation_t opX,
                                                aclsparseOperation_t opY) {
    std::vector<double> goldenX = Xf64;
    std::vector<double> goldenY = Yf64;
    SddmmCsr goldenCsr = csrC;
    if (dtype == ACL_FLOAT16) {
        goldenX = QuantizeDoublesViaFp16(Xf64);
        goldenY = QuantizeDoublesViaFp16(Yf64);
        goldenCsr.values = QuantizeDoublesViaFp16(csrC.values);
    }
    return SddmmGolden(p.m, p.n, p.k, goldenX, goldenY, goldenCsr,
                       p.alpha, p.beta, opX, opY);
}

template <typename T>
static void RepackSddmmDenseInput(std::vector<T> &input,
    aclsparseOrder_t order, int64_t rows, int64_t cols)
{
    if (order == ACL_SPARSE_ORDER_COL) {
        input = RepackRowToCol(input, rows, cols);
    }
}

SddmmNpuResult RunNpuSddmm(const SddmmTestParam& p, aclrtStream stream,
                                   const SddmmCsr& csrC,
                                   const std::vector<double>& Xf64,
                                   const std::vector<double>& Yf64,
                                   aclDataType dtype, aclDataType computeType,
                                   aclsparseSDDMMAlg_t alg,
                                   aclsparseOperation_t opX, aclsparseOperation_t opY,
                                   aclsparseOrder_t orderX, aclsparseOrder_t orderY,
                                   int64_t nnz) {
    HandleManager handle;
    float alphaF = static_cast<float>(p.alpha);
    float betaF = static_cast<float>(p.beta);
    const bool xTransposed = (opX == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t xRows = xTransposed ? p.k : p.m;
    const int64_t xCols = xTransposed ? p.m : p.k;
    const bool yTransposed = (opY == ACL_SPARSE_OP_TRANSPOSE);
    const int64_t yRows = yTransposed ? p.n : p.k;
    const int64_t yCols = yTransposed ? p.k : p.n;

    SddmmNpuResult npuResult;
    if (dtype == ACL_FLOAT) {
        auto hCInit = DoublesToFp32(csrC.values);
        auto hX = DoublesToFp32(Xf64);
        auto hY = DoublesToFp32(Yf64);
        RepackSddmmDenseInput(hX, orderX, xRows, xCols);
        RepackSddmmDenseInput(hY, orderY, yRows, yCols);
        npuResult = SddmmNpu<float>(handle, stream, p.m, p.n, p.k,
                                     opX, opY, alphaF, betaF,
                                     dtype, computeType, alg, orderX, orderY,
                                     csrC.rowOffsets, csrC.colIndices,
                                     hCInit, hX, hY, nnz);
    } else {
        auto hCInit = DoublesToFp16(csrC.values);
        auto hX = DoublesToFp16(Xf64);
        auto hY = DoublesToFp16(Yf64);
        RepackSddmmDenseInput(hX, orderX, xRows, xCols);
        RepackSddmmDenseInput(hY, orderY, yRows, yCols);
        npuResult = SddmmNpu<uint16_t>(handle, stream, p.m, p.n, p.k,
                                        opX, opY, alphaF, betaF,
                                        dtype, computeType, alg, orderX, orderY,
                                        csrC.rowOffsets, csrC.colIndices,
                                        hCInit, hX, hY, nnz);
    }
    return npuResult;
}

void VerifySddmmPrecision(const std::string& caseName,
                                  const SddmmNpuResult& npuResult,
                                  const std::vector<double>& expectFp64,
                                  aclDataType dtype, int64_t nnz) {
    if (nnz <= 0) {
        std::cout << "[" << caseName << "] nnz=0 (empty tensor), skip value verify\n";
        return;
    }
    std::vector<float> goldenFloat(static_cast<size_t>(nnz));
    std::vector<float> npuFloat(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; i++) {
        float gv = static_cast<float>(expectFp64[static_cast<size_t>(i)]);
        if (dtype == ACL_FLOAT16) {
            gv = Fp16BitsToFp32(Fp32ToFp16Bits(gv));
        }
        goldenFloat[static_cast<size_t>(i)] = gv;
        npuFloat[static_cast<size_t>(i)] =
            static_cast<float>(npuResult.valuesOut[static_cast<size_t>(i)]);
    }
    // Framework MIXED_TOLERANCE mode: dtype-driven atol/rtol + per-element
    // max(abs_err) limit = max(fixedValue, 32*ULP), requiredMatchedRatio=0.99.
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, goldenFloat.data(), goldenFloat.size());
    bool pass = Verifier::verifyVector(npuFloat.data(), goldenFloat.data(),
                                       static_cast<size_t>(nnz), 1, cfg, caseName);
    EXPECT_TRUE(pass) << "MIXED_TOLERANCE verification failed for " << caseName;
}

void CheckDenseBatchStrideRequiresFullPaddedSpan(
    HandleManager *handle_, float &alpha_, float &beta_, DeviceBuffer &dBuffer_)
{
    const std::vector<float> xValues(24, 1.0F);
    const std::vector<float> yValues(6, 1.0F);
    const std::vector<float> cValues(8, 0.0F);
    const int32_t offsets[] = {0, 1};
    const int32_t columns[] = {0};
    auto x = DeviceBuffer::copyFrom(xValues.data(), xValues.size() * sizeof(float));
    auto y = DeviceBuffer::copyFrom(yValues.data(), yValues.size() * sizeof(float));
    auto c = DeviceBuffer::copyFrom(cValues.data(), cValues.size() * sizeof(float));
    auto rowOffsets = DeviceBuffer::copyFrom(offsets, sizeof(offsets));
    auto colIndices = DeviceBuffer::copyFrom(columns, sizeof(columns));
    auto matY = DnMatManager::createConst(3, 2, 2, y.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    BsrCreateParams p;
    p.blockRows = p.blockCols = p.blockNnz = 1;
    p.rowOffsets = rowOffsets.raw();
    p.colIndices = colIndices.raw();
    p.values = c.raw();
    MutableSpMatGuard matC;
    ASSERT_EQ(CreateMutableBsr(p, matC.put()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseBsrSetStridedBatch(matC.get(), 2, 0, 0, 4), ACL_SPARSE_STATUS_SUCCESS);
    for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
        auto matX = DnMatManager::createConst(2, 3, 4, x.raw(), ACL_FLOAT, order);
        const int64_t readableSpan = order == ACL_SPARSE_ORDER_ROW ? 7 : 10;
        const int64_t paddedSpan = order == ACL_SPARSE_ORDER_ROW ? 8 : 12;
        ASSERT_EQ(aclsparseDnMatSetStridedBatch(matX.get(), 2, readableSpan), ACL_SPARSE_STATUS_SUCCESS);
        size_t size = 0;
        EXPECT_EQ(aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_INVALID_VALUE);
        EXPECT_EQ(aclsparseSDDMMPreprocess(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer_.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
        ASSERT_EQ(aclsparseDnMatSetStridedBatch(matX.get(), 2, paddedSpan), ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX.cget(), matY.cget(), &beta_, matC.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_SUCCESS);
    }
}

void CheckZeroOutputDimension(aclrtStream stream, bool zeroRows)
{
    const int64_t m = zeroRows ? 0 : 4;
    const int64_t n = zeroRows ? 4 : 0;
    const std::vector<int32_t> offsets(static_cast<size_t>(m + 1), 0);
    const std::vector<float> dense(16, 1.0F);
    const int32_t column = 0;
    auto dOffsets = DeviceBuffer::copyFrom(offsets.data(), offsets.size() * sizeof(int32_t));
    auto dDense = DeviceBuffer::copyFrom(dense.data(), dense.size() * sizeof(float));
    auto dColumn = DeviceBuffer::copyFrom(&column, sizeof(column));
    auto matX = DnMatManager::createConst(
        m, 4, 4, zeroRows ? nullptr : dDense.get(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(
        4, n, zeroRows ? 4 : 1, zeroRows ? dDense.get() : nullptr, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(m, n, 0, dOffsets.get(), nullptr, nullptr);
    HandleManager handle;
    handle.setStream(stream);
    const float alpha = 1.0F;
    const float beta = -0.5F;
    DeviceBuffer workspace;
    const auto stages = sddmm_extended::RunThreeStages(
        handle, &alpha, &beta, matX, matY, matC.get(), ACL_FLOAT, stream, workspace);
    ASSERT_TRUE(sddmm_extended::StagesSucceeded(stages));
    EXPECT_EQ(stages.workspaceSize, 0U);
    EXPECT_EQ(workspace.get(), nullptr);

    // CreateCsr permits delayed binding; SDDMM rejects a nonempty zero-size C.
    auto nonempty = SpMatManager::createCsr(m, n, 1, dOffsets.get(), dColumn.get(), dDense.get());
    size_t rejectedSize = 0;
    EXPECT_EQ(aclsparseSDDMMBufferSize(
        handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha, matX.cget(), matY.cget(), &beta, nonempty.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &rejectedSize), ACL_SPARSE_STATUS_INVALID_VALUE);
}

void CheckZeroReduction(aclrtStream stream, float beta)
{
    SddmmCsr pattern;
    pattern.m = 2;
    pattern.n = 3;
    pattern.nnz = 3;
    pattern.rowOffsets = {0, 1, 3};
    pattern.colIndices = {0, 1, 2};
    pattern.values = {0.25, -0.5, 1.0};
    auto initial = DoublesToFp32(pattern.values);
    if (beta == 0.0F) {
        std::fill(initial.begin(), initial.end(), std::numeric_limits<float>::quiet_NaN());
    }
    auto dOffsets = DeviceBuffer::copyFrom(
        pattern.rowOffsets.data(), pattern.rowOffsets.size() * sizeof(int32_t));
    auto dColumns = DeviceBuffer::copyFrom(
        pattern.colIndices.data(), pattern.colIndices.size() * sizeof(int32_t));
    auto dValues = DeviceBuffer::copyFrom(initial.data(), initial.size() * sizeof(float));
    auto matX = DnMatManager::createConst(2, 0, 1, nullptr, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matY = DnMatManager::createConst(0, 3, 3, nullptr, ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    auto matC = SpMatManager::createCsr(2, 3, pattern.nnz, dOffsets.get(), dColumns.get(), dValues.get());
    HandleManager handle;
    handle.setStream(stream);
    const float alpha = 2.0F;
    DeviceBuffer workspace;
    const auto stages = sddmm_extended::RunThreeStages(
        handle, &alpha, &beta, matX, matY, matC.get(), ACL_FLOAT, stream, workspace);
    ASSERT_TRUE(sddmm_extended::StagesSucceeded(stages));
    std::vector<float> actual(initial.size());
    dValues.copyToHost(actual.data(), actual.size() * sizeof(float));
    const auto expected = beta == 0.0F ? std::vector<float>(initial.size(), 0.0F) :
        DoublesToFp32(SddmmGolden(2, 3, 0, {}, {}, pattern, alpha, beta));
    VerifyConfig config;
    config.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(actual, expected, config, "SDDMM_K_ZERO"));
}

aclsparseStatus_t CreateMutableBsr(
    const BsrCreateParams& p, aclsparseSpMatDescr_t* descr) {
    return aclsparseCreateBsr(
        descr, p.blockRows, p.blockCols, p.blockNnz,
        p.rowBlockSize, p.colBlockSize, p.rowOffsets, p.colIndices,
        p.values, p.rowOffsetsType, p.colIndicesType, p.indexBase,
        p.valueType, p.order);
}

aclsparseStatus_t CreateConstBsr(
    const BsrCreateParams& p, aclsparseConstSpMatDescr_t* descr) {
    return aclsparseCreateConstBsr(
        descr, p.blockRows, p.blockCols, p.blockNnz,
        p.rowBlockSize, p.colBlockSize, p.rowOffsets, p.colIndices,
        p.values, p.rowOffsetsType, p.colIndicesType, p.indexBase,
        p.valueType, p.order);
}

void ExpectMutableAndConstBsrCreate(
    const BsrCreateParams& p, aclsparseStatus_t expected) {
    MutableSpMatGuard mutableDescr;
    ConstSpMatGuard constDescr;

    EXPECT_EQ(CreateMutableBsr(p, mutableDescr.put()), expected);
    EXPECT_EQ(CreateConstBsr(p, constDescr.put()), expected);
    if (expected == ACL_SPARSE_STATUS_SUCCESS) {
        EXPECT_NE(mutableDescr.get(), nullptr);
        EXPECT_NE(constDescr.get(), nullptr);

        aclsparseFormat_t mutableFormat{};
        aclsparseFormat_t constFormat{};
        int64_t mutableRows = -1;
        int64_t mutableCols = -1;
        int64_t mutableNnz = -1;
        int64_t constRows = -1;
        int64_t constCols = -1;
        int64_t constNnz = -1;
        void* mutableValues = nullptr;
        const void* constValues = nullptr;
        EXPECT_EQ(aclsparseSpMatGetFormat(
                      mutableDescr.get(), &mutableFormat),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseSpMatGetFormat(
                      constDescr.get(), &constFormat),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseSpMatGetSize(
                      mutableDescr.get(), &mutableRows, &mutableCols,
                      &mutableNnz),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseSpMatGetSize(
                      constDescr.get(), &constRows, &constCols, &constNnz),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseSpMatGetValues(
                      mutableDescr.get(), &mutableValues),
                  ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseConstSpMatGetValues(
                      constDescr.get(), &constValues),
                  ACL_SPARSE_STATUS_SUCCESS);

        EXPECT_EQ(mutableFormat, ACL_SPARSE_FORMAT_BSR);
        EXPECT_EQ(constFormat, ACL_SPARSE_FORMAT_BSR);
        EXPECT_EQ(mutableRows, p.blockRows * p.rowBlockSize);
        EXPECT_EQ(mutableCols, p.blockCols * p.colBlockSize);
        EXPECT_EQ(mutableNnz, p.blockNnz);
        EXPECT_EQ(constRows, mutableRows);
        EXPECT_EQ(constCols, mutableCols);
        EXPECT_EQ(constNnz, mutableNnz);
        EXPECT_EQ(mutableValues, p.values);
        EXPECT_EQ(constValues, p.values);
    }
}

void ExpectDnMatBatchState(
    aclsparseConstDnMatDescr_t descr, int expectedCount,
    int64_t expectedStride) {
    int batchCount = -1;
    int64_t batchStride = -1;
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(
                  descr, &batchCount, &batchStride),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(batchCount, expectedCount);
    EXPECT_EQ(batchStride, expectedStride);
}

CsrWorkspaceFixture::CsrWorkspaceFixture(aclrtStream callStream, CsrWorkspacePattern pattern,
    int64_t rows, aclsparseIndexBase_t base,
    int32_t lastRowEntries)
    : stream(callStream), rowCount(rows), indexBase(base), lastRowNnz(lastRowEntries)
{
    if (rows <= 0 || lastRowEntries <= 0 || lastRowEntries > 64) {
        throw std::invalid_argument("invalid CSR workspace test shape");
    }
    SetHostTestStream(handle, stream);
    MakePattern(pattern);
    PrepareInputs();
    CreateDescriptors();
}

bool CsrWorkspaceFixture::Query()
{
    const auto before = dValues.snapshot();
    if (!QueryTestWorkspace(stages, handle, Alpha(), Beta(), matX, matY,
            matC.get(), ACL_FLOAT)) {
        return false;
    }
    stages.synchronizeRet = aclrtSynchronizeStream(stream);
    if (stages.synchronizeRet != ACL_SUCCESS) {
        return false;
    }
    EXPECT_TRUE(stages.workspaceSizeAligned);
    EXPECT_EQ(before, dValues.snapshot()) << "BufferSize must preserve C";
    CheckStorage();
    return true;
}

void CsrWorkspaceFixture::AllocateWorkspace()
{
    workspace = SddmmGuardedBuffer::copyFrom(nullptr, stages.workspaceSize);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(workspace.get()) %
        kSddmmWorkspaceAlignment, 0U);
}

aclsparseStatus_t CsrWorkspaceFixture::Preprocess()
{
    return aclsparseSDDMMPreprocess(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, Alpha(), matX.cget(), matY.cget(),
        Beta(), matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace.get());
}

aclsparseStatus_t CsrWorkspaceFixture::Execute()
{
    return aclsparseSDDMM(handle.get(), ACL_SPARSE_OP_NON_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, Alpha(), matX.cget(), matY.cget(),
        Beta(), matC.get(), ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, workspace.get());
}

bool CsrWorkspaceFixture::Run(bool preprocess)
{
    dValues.restore();
    if (preprocess && !PrepareAndCheck()) {
        return false;
    }
    stages.executeRet = Execute();
    if (stages.executeRet != ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    stages.synchronizeRet = aclrtSynchronizeStream(stream);
    if (stages.synchronizeRet != ACL_SUCCESS) {
        return false;
    }
    CheckStorage();
    RealRunResult result;
    result.actual = UnpackRealValues(dValues.snapshot(), columns.size(), ACL_FLOAT);
    result.golden = Golden();
    return VerifyRealRun(result, ACL_FLOAT, "CSR_WORKSPACE_PLAN");
}

void CsrWorkspaceFixture::ChangePattern(CsrWorkspacePattern pattern, bool replacePointer)
{
    MakePattern(pattern);
    UploadIndices(dOffsets, offsets);
    const size_t bytes = columns.size() * sizeof(int32_t);
    if (replacePointer) {
        auto replacement = SddmmGuardedBuffer::copyFrom(columns.data(), bytes);
        CheckSetupStatus(aclsparseCsrSetPointers(matC.get(), dOffsets.get(),
            replacement.get(), dValues.get()), "aclsparseCsrSetPointers");
        dColumns = std::move(replacement);
    } else if (aclrtMemcpy(dColumns.get(), bytes, columns.data(), bytes,
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("workspace test pattern H2D failed");
    }
}

void CsrWorkspaceFixture::PerturbRow(int64_t row)
{
    if (row < 0 || row >= rowCount || lastRowNnz != 64) {
        throw std::invalid_argument("invalid CSR workspace perturbation row");
    }
    auto begin = columns.begin() + row * 64;
    begin[31] = static_cast<int32_t>((64 + row * 17 + 96) % n) + indexBase;
    std::sort(begin, begin + 64);
    UploadIndices(dColumns, columns);
}

void CsrWorkspaceFixture::ShiftRows(int64_t firstRow, int32_t columnShift)
{
    if (firstRow < 0 || firstRow >= rowCount || lastRowNnz != 64 ||
        columnShift < 0 || columnShift >= n) {
        throw std::invalid_argument("invalid CSR workspace row shift");
    }
    for (int64_t row = firstRow; row < rowCount; ++row) {
        auto begin = columns.begin() + row * 64;
        for (int32_t slot = 0; slot < 64; ++slot) {
            begin[slot] = (begin[slot] - indexBase + columnShift) % n + indexBase;
        }
        std::sort(begin, begin + 64);
    }
    UploadIndices(dColumns, columns);
}

int32_t CsrWorkspaceFixture::Column(size_t position) const
{ return columns.at(position); }

int32_t CsrWorkspaceFixture::Offset(size_t position) const
{ return offsets.at(position); }

size_t CsrWorkspaceFixture::ColumnCount() const
{ return columns.size(); }

void CsrWorkspaceFixture::SetColumn(size_t position, int32_t value)
{
    columns.at(position) = value;
    UploadIndices(dColumns, columns);
}

void CsrWorkspaceFixture::SetOffset(size_t position, int32_t value)
{
    offsets.at(position) = value;
    UploadIndices(dOffsets, offsets);
}

bool CsrWorkspaceFixture::Prepare()
{ return PrepareAndCheck(); }

bool CsrWorkspaceFixture::IsPrepared() const
{ return matC.cget()->sddmmPrepared.prepared; }

uint32_t CsrWorkspaceFixture::PatternFlags() const
{ return matC.cget()->sddmmPrepared.patternFlags; }

void CsrWorkspaceFixture::CheckRejectedPattern()
{
    const auto before = dValues.snapshot();
    ASSERT_EQ(Preprocess(), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_FALSE(IsPrepared());
    ASSERT_EQ(Execute(), ACL_SPARSE_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_FALSE(IsPrepared());
    EXPECT_EQ(before, dValues.snapshot()) << "invalid pattern must preserve C";
    CheckStorage();
}

void CsrWorkspaceFixture::SetDeviceScalars(bool deviceMode)
{
    CheckSetupStatus(aclsparseSetPointerMode(handle.get(), deviceMode ?
        ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST),
        "aclsparseSetPointerMode");
    deviceScalars = deviceMode;
    if (deviceMode) {
        dAlpha = DeviceBuffer::copyFrom(&alpha, sizeof(alpha));
        dBeta = DeviceBuffer::copyFrom(&beta, sizeof(beta));
    }
}

void CsrWorkspaceFixture::ChangeYToColumnMajor()
{
    const auto packed = PackRealValues(PackColumnMajor(y, k, n), ACL_FLOAT16);
    dY = SddmmGuardedBuffer::copyFrom(packed.data(), packed.size());
    matY = DnMatManager::createConst(k, n, k, dY.get(),
        ACL_FLOAT16, ACL_SPARSE_ORDER_COL);
}

void CsrWorkspaceFixture::CheckStorage() const
{
    CheckGuardedBuffers(dValues, workspace, {&dX, &dY});
    EXPECT_TRUE(dOffsets.guardsIntact()) << "offset canary";
    std::vector<int32_t> actualOffsets(offsets.size());
    dOffsets.copyToHost(actualOffsets.data(), actualOffsets.size() * sizeof(int32_t));
    EXPECT_EQ(actualOffsets, offsets) << "readonly row offsets";
    EXPECT_TRUE(dColumns.guardsIntact()) << "column canary";
    std::vector<int32_t> actual(columns.size());
    dColumns.copyToHost(actual.data(), actual.size() * sizeof(int32_t));
    EXPECT_EQ(actual, columns) << "readonly column indices";
}

const void *CsrWorkspaceFixture::Alpha() const
{ return deviceScalars ? dAlpha.raw() : &alpha; }

const void *CsrWorkspaceFixture::Beta() const
{ return deviceScalars ? dBeta.raw() : &beta; }

void CsrWorkspaceFixture::MakePattern(CsrWorkspacePattern pattern)
{
    const int64_t stride = pattern == CsrWorkspacePattern::WEAK ? 19 : 17;
    const int64_t first = pattern == CsrWorkspacePattern::UNALIGNED ? 65 :
        (pattern == CsrWorkspacePattern::ALIGNED_SHIFTED ? 80 : 64);
    MakeCyclicCsrPattern(rowCount, n, stride, first, static_cast<int32_t>(indexBase),
        pattern == CsrWorkspacePattern::PERTURBED, offsets, columns);
    columns.resize(columns.size() - static_cast<size_t>(64 - lastRowNnz));
    offsets.back() = static_cast<int32_t>(columns.size()) + indexBase;
}

void CsrWorkspaceFixture::UploadIndices(SddmmGuardedBuffer &device, const std::vector<int32_t> &host)
{
    const size_t bytes = host.size() * sizeof(int32_t);
    if (aclrtMemcpy(device.get(), bytes, host.data(), bytes,
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("workspace test index H2D failed");
    }
}

void CsrWorkspaceFixture::PrepareInputs()
{
    x = QuantizeRealVector(MakeRealValues(1, rowCount * k, 0.0625F), ACL_FLOAT16);
    y = QuantizeRealVector(MakeRealValues(1, k * n, -0.03125F), ACL_FLOAT16);
    cInitial = MakeRealValues(1, static_cast<int64_t>(columns.size()), 0.015625F);
    const auto xBytes = PackRealValues(x, ACL_FLOAT16);
    const auto yBytes = PackRealValues(y, ACL_FLOAT16);
    dOffsets = SddmmGuardedBuffer::copyFrom(
        offsets.data(), offsets.size() * sizeof(int32_t));
    dColumns = SddmmGuardedBuffer::copyFrom(
        columns.data(), columns.size() * sizeof(int32_t));
    dX = SddmmGuardedBuffer::copyFrom(xBytes.data(), xBytes.size());
    dY = SddmmGuardedBuffer::copyFrom(yBytes.data(), yBytes.size());
    dValues = SddmmGuardedBuffer::copyFrom(
        cInitial.data(), cInitial.size() * sizeof(float));
}

void CsrWorkspaceFixture::CreateDescriptors()
{
    matX = DnMatManager::createConst(rowCount, k, k, dX.get(), ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    matY = DnMatManager::createConst(k, n, n, dY.get(), ACL_FLOAT16, ACL_SPARSE_ORDER_ROW);
    matC = SpMatManager::createCsr(rowCount, n, static_cast<int64_t>(columns.size()),
        dOffsets.get(), dColumns.get(), dValues.get(), ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_32I, indexBase, ACL_FLOAT);
}

bool CsrWorkspaceFixture::PrepareAndCheck()
{
    const auto before = dValues.snapshot();
    stages.preprocessRet = Preprocess();
    if (stages.preprocessRet != ACL_SPARSE_STATUS_SUCCESS) {
        return false;
    }
    stages.synchronizeRet = aclrtSynchronizeStream(stream);
    if (stages.synchronizeRet != ACL_SUCCESS) {
        return false;
    }
    EXPECT_EQ(before, dValues.snapshot()) << "Preprocess must preserve C";
    CheckStorage();
    return true;
}

std::vector<float> CsrWorkspaceFixture::Golden() const
{
    SddmmCsr csr;
    csr.m = rowCount;
    csr.n = n;
    csr.nnz = static_cast<int64_t>(columns.size());
    csr.rowOffsets = offsets;
    csr.colIndices = columns;
    for (auto &offset : csr.rowOffsets) {
        offset -= indexBase;
    }
    for (auto &column : csr.colIndices) {
        column -= indexBase;
    }
    csr.values.assign(cInitial.begin(), cInitial.end());
    const std::vector<double> doubleX(x.begin(), x.end());
    const std::vector<double> doubleY(y.begin(), y.end());
    return DoublesToFp32(SddmmGolden(rowCount, n, k, doubleX, doubleY, csr, alpha, beta));
}
} // namespace sddmm_extended
} // namespace sparse_test
