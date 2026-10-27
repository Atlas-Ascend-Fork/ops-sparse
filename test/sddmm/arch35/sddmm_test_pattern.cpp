/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_common.h"
#include "../../../sparse/sddmm/arch35/host/sddmm_pattern_host.h"
#include "../../../sparse/sddmm/arch35/sddmm_kernel.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace sparse_test {
namespace sddmm_pattern {
namespace {

constexpr size_t kGuardBytes = 64U;
constexpr size_t kAllocationSlack = 1024U;
constexpr uint8_t kGuardValue = 0xA5U;
constexpr int64_t kCyclicColumns = 64;

struct Pattern {
    SddmmTilingData tiling{};
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
};

struct ExpectedPattern {
    int32_t status = SDDMM_PATTERN_VALID;
    uint32_t flags = SDDMM_PATTERN_FLAG_NONE;
    int32_t firstColumn = -1;
    int64_t firstInvalidBatch = -1;
};

Pattern MakeCsr(const std::vector<int32_t> &degrees, int32_t columns, int32_t base)
{
    if (columns == 0) {
        ADD_FAILURE() << "CSR test columns must be nonzero";
        return {};
    }
    if (columns < 0) {
        throw std::invalid_argument("CSR test columns must be positive");
    }
    Pattern pattern;
    auto &tiling = pattern.tiling;
    tiling.format = SDDMM_FORMAT_CSR;
    tiling.m = static_cast<int64_t>(degrees.size());
    tiling.n = columns;
    tiling.indexBase = base;
    tiling.batchCount = 1;
    pattern.offsets.push_back(base);
    for (const int32_t degree : degrees) {
        for (int32_t index = 0; index < degree; ++index) {
            pattern.columns.push_back(base + (index * 7 + degree) % columns);
        }
        pattern.offsets.push_back(static_cast<int32_t>(pattern.columns.size()) + base);
    }
    tiling.nnzUnits = static_cast<int64_t>(pattern.columns.size());
    return pattern;
}

Pattern MakeCyclic(int32_t rows, int32_t columns, int32_t base, int32_t rowStride)
{
    if (columns == 0) {
        ADD_FAILURE() << "Cyclic test columns must be nonzero";
        return {};
    }
    if (columns < 0) {
        throw std::invalid_argument("Cyclic test columns must be positive");
    }
    auto pattern = MakeCsr(std::vector<int32_t>(rows, kCyclicColumns), columns, base);
    for (int32_t row = 0; row < rows; ++row) {
        auto begin = pattern.columns.begin() + row * kCyclicColumns;
        const int32_t start = (11 + rowStride * row) % columns;
        for (int32_t index = 0; index < kCyclicColumns; ++index) {
            begin[index] = base + (start + index) % columns;
        }
        std::sort(begin, begin + kCyclicColumns);
    }
    return pattern;
}

Pattern MakeBsr(bool independentOffsets, bool independentColumns, int32_t base)
{
    Pattern pattern;
    auto &tiling = pattern.tiling;
    tiling.format = SDDMM_FORMAT_BSR;
    tiling.blockRows = 5;
    tiling.blockCols = 97;
    tiling.blockDim = 4;
    tiling.m = tiling.blockRows * tiling.blockDim;
    tiling.n = tiling.blockCols * tiling.blockDim;
    tiling.nnzUnits = 13;
    tiling.indexBase = base;
    tiling.batchCount = 3;
    tiling.offsetsBatchStride = independentOffsets ? 9 : 0;
    tiling.columnsBatchStride = independentColumns ? 18 : 0;
    const int64_t offsetsCount = independentOffsets ? tiling.batchCount : 1;
    const int64_t columnsCount = independentColumns ? tiling.batchCount : 1;
    pattern.offsets.assign((offsetsCount - 1) * tiling.offsetsBatchStride + 6, -12345);
    pattern.columns.assign((columnsCount - 1) * tiling.columnsBatchStride + tiling.nnzUnits, -12345);
    for (int64_t batch = 0; batch < offsetsCount; ++batch) {
        const int64_t offset = batch * tiling.offsetsBatchStride;
        const int32_t firstDegree = 2 + static_cast<int32_t>(batch % 2);
        const std::vector<int32_t> degrees = {firstDegree, 0, 5, 1, 7 - firstDegree};
        pattern.offsets[offset] = base;
        for (size_t row = 0; row < degrees.size(); ++row) {
            pattern.offsets[offset + row + 1] = pattern.offsets[offset + row] + degrees[row];
        }
    }
    for (int64_t batch = 0; batch < columnsCount; ++batch) {
        for (int64_t index = 0; index < tiling.nnzUnits; ++index) {
            pattern.columns[batch * tiling.columnsBatchStride + index] =
                base + static_cast<int32_t>((index * 7 + batch * 13) % tiling.blockCols);
        }
    }
    return pattern;
}

// Independently enumerate candidate cyclic sets instead of reproducing the
// Device gap-recovery implementation or its partitioning.
int32_t OracleCyclicStart(const std::vector<int32_t> &stored, int32_t columns, int32_t base)
{
    if (columns == 0) {
        return -1;
    }
    if (columns < kCyclicColumns || stored.size() != static_cast<size_t>(kCyclicColumns) ||
        std::adjacent_find(stored.begin(), stored.end(), std::greater_equal<int32_t>()) != stored.end()) {
        return -1;
    }
    for (int32_t start = 0; start < columns; ++start) {
        std::vector<int32_t> expected(kCyclicColumns);
        for (int32_t index = 0; index < kCyclicColumns; ++index) {
            expected[index] = base + (start + index) % columns;
        }
        std::sort(expected.begin(), expected.end());
        if (stored == expected) {
            return start;
        }
    }
    return -1;
}

int32_t OracleBatchStatus(const Pattern &pattern, int64_t batch)
{
    const auto &tiling = pattern.tiling;
    const int64_t rows = tiling.format == SDDMM_FORMAT_CSR ? tiling.m : tiling.blockRows;
    const int64_t columns = tiling.format == SDDMM_FORMAT_CSR ? tiling.n : tiling.blockCols;
    const int64_t offsetsBase = batch * tiling.offsetsBatchStride;
    const int64_t columnsBase = batch * tiling.columnsBatchStride;
    if (pattern.offsets[offsetsBase] != tiling.indexBase ||
        pattern.offsets[offsetsBase + rows] != tiling.indexBase + tiling.nnzUnits) {
        return SDDMM_PATTERN_BAD_ENDPOINT;
    }
    for (int64_t row = 0; row < rows; ++row) {
        const int32_t current = pattern.offsets[offsetsBase + row];
        const int32_t next = pattern.offsets[offsetsBase + row + 1];
        if (current < tiling.indexBase || next < current || next > tiling.indexBase + tiling.nnzUnits) {
            return SDDMM_PATTERN_NOT_MONOTONIC;
        }
    }
    for (int64_t index = 0; index < tiling.nnzUnits; ++index) {
        const int32_t column = pattern.columns[columnsBase + index];
        if (column < tiling.indexBase || column >= columns + tiling.indexBase) {
            return SDDMM_PATTERN_BAD_COLUMN;
        }
    }
    return SDDMM_PATTERN_VALID;
}

ExpectedPattern SerialOracle(const Pattern &pattern)
{
    const auto &tiling = pattern.tiling;
    const int64_t rows = tiling.format == SDDMM_FORMAT_CSR ? tiling.m : tiling.blockRows;
    const int64_t columns = tiling.format == SDDMM_FORMAT_CSR ? tiling.n : tiling.blockCols;
    const int64_t batches = tiling.offsetsBatchStride == 0 && tiling.columnsBatchStride == 0 ?
        1 : tiling.batchCount;
    ExpectedPattern expected;
    for (int64_t batch = 0; batch < batches; ++batch) {
        const int32_t status = OracleBatchStatus(pattern, batch);
        if (status != SDDMM_PATTERN_VALID) {
            expected.status = status;
            expected.firstInvalidBatch = batch;
            return expected;
        }
    }
    if (tiling.format != SDDMM_FORMAT_CSR || batches != 1 || rows == 0 || columns < kCyclicColumns) {
        return expected;
    }
    std::vector<int32_t> starts;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t begin = pattern.offsets[row] - tiling.indexBase;
        const int64_t end = pattern.offsets[row + 1] - tiling.indexBase;
        if (end - begin != kCyclicColumns) {
            return expected;
        }
        const std::vector<int32_t> stored(pattern.columns.begin() + begin, pattern.columns.begin() + end);
        const int32_t start = OracleCyclicStart(stored, static_cast<int32_t>(columns), tiling.indexBase);
        if (start < 0) {
            return expected;
        }
        starts.push_back(start);
    }
    expected.flags = SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64;
    expected.firstColumn = starts.front();
    bool stride17 = true;
    for (int64_t row = 0; row < rows; ++row) {
        stride17 = stride17 && starts[row] == (starts.front() + 17 * row) % columns;
    }
    if (stride17) {
        expected.flags |= SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17;
    }
    return expected;
}

struct DevicePattern {
    DeviceBuffer offsets;
    DeviceBuffer columns;
    DeviceBuffer workspace;
    size_t partialBytes = 0;
};

void Upload(const std::vector<int32_t> &values, DeviceBuffer &buffer)
{
    if (values.empty()) {
        return;
    }
    const size_t bytes = values.size() * sizeof(int32_t);
    buffer = DeviceBuffer::alloc(bytes + kAllocationSlack);
    ASSERT_EQ(aclrtMemcpy(buffer.get(), buffer.size(), values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);
}

void Prepare(const Pattern &pattern, uint32_t cores, DevicePattern &device)
{
    ASSERT_NO_FATAL_FAILURE(Upload(pattern.offsets, device.offsets));
    ASSERT_NO_FATAL_FAILURE(Upload(pattern.columns, device.columns));
    device.partialBytes = cores * sizeof(SddmmQueryPartial);
    device.workspace = DeviceBuffer::alloc(device.partialBytes + kGuardBytes + kAllocationSlack);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(device.workspace.get()) % kSddmmWorkspaceAlignment, 0U);
    ASSERT_EQ(aclrtMemset(device.workspace.get(), device.workspace.size(), kGuardValue,
        device.workspace.size()), ACL_SUCCESS);
}

void Launch(DevicePattern &device, const Pattern &pattern, uint32_t cores, aclrtStream stream)
{
    sddmm_pattern_validate_kernel_do(reinterpret_cast<GM_ADDR>(device.offsets.get()),
        reinterpret_cast<GM_ADDR>(device.columns.get()), reinterpret_cast<GM_ADDR>(device.workspace.get()),
        pattern.tiling, cores, stream);
}

void CheckUnchangedInput(const std::vector<int32_t> &expected, const DeviceBuffer &device)
{
    std::vector<int32_t> actual(expected.size());
    device.copyToHost(actual.data(), actual.size() * sizeof(int32_t));
    EXPECT_EQ(actual, expected);
}

void CheckResult(const Pattern &pattern, uint32_t cores, const DevicePattern &device)
{
    std::vector<uint8_t> bytes(device.partialBytes + kGuardBytes);
    device.workspace.copyToHost(bytes.data(), bytes.size());
    std::vector<SddmmQueryPartial> partials(cores);
    ASSERT_EQ(device.partialBytes, partials.size() * sizeof(SddmmQueryPartial));
    std::copy_n(bytes.data(), device.partialBytes, reinterpret_cast<uint8_t *>(partials.data()));
    SddmmValidationResult reduced{};
    ASSERT_TRUE(sddmm_arch35::ReducePatternValidation(partials.data(), cores, pattern.tiling, reduced));
    const auto expected = SerialOracle(pattern);
    EXPECT_EQ(reduced.magic, kSddmmValidationMagic);
    EXPECT_EQ(reduced.reserved, 0);
    EXPECT_EQ(reduced.status, expected.status);
    EXPECT_EQ(reduced.patternFlags, expected.flags);
    EXPECT_EQ(reduced.logicalFirstColumn, expected.firstColumn);
    int64_t earliestBatch = -1;
    for (const auto &partial : partials) {
        if (partial.firstInvalidBatch >= 0 &&
            (earliestBatch < 0 || partial.firstInvalidBatch < earliestBatch)) {
            earliestBatch = partial.firstInvalidBatch;
        }
    }
    EXPECT_EQ(earliestBatch, expected.firstInvalidBatch);
    EXPECT_TRUE(std::all_of(bytes.begin() + device.partialBytes, bytes.end(),
        [](uint8_t value) { return value == kGuardValue; }));
    ASSERT_NO_FATAL_FAILURE(CheckUnchangedInput(pattern.offsets, device.offsets));
    ASSERT_NO_FATAL_FAILURE(CheckUnchangedInput(pattern.columns, device.columns));
}

struct TimingEvents {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ~TimingEvents()
    {
        if (start != nullptr) {
            (void)aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            (void)aclrtDestroyEvent(stop);
        }
    }
};

void PrintTimingSamples(const Pattern &pattern, uint32_t cores, const std::vector<double> &samples,
    int warmup, int launchesPerRound, int rounds)
{
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    std::cout << "[sddmm-pattern-event] scope=kernel_only same_candidate=true rows=" << pattern.tiling.m
              << " columns=" << pattern.tiling.n << " nnz_units=" << pattern.tiling.nnzUnits
              << " input_bytes=" << (pattern.offsets.size() + pattern.columns.size()) * sizeof(int32_t)
              << " cores=" << cores << " warmup=" << warmup << " launches=" << launchesPerRound
              << " rounds=" << rounds << " samples_us=";
    for (const double sample : samples) {
        std::cout << sample << ',';
    }
    std::cout << " median_us=" << sorted[sorted.size() / 2] << '\n';
}

class SddmmPatternNpuTest : public testing::Test {
public:
    static void SetUpTestSuite()
    {
        env_ = std::make_unique<AclEnvScope>();
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        ASSERT_NE(platform, nullptr);
        runtimeCores_ = platform->GetCoreNumAiv();
        ASSERT_GE(runtimeCores_, 2U);
    }

    static void TearDownTestSuite()
    {
        if (failedSynchronization_) {
            (void)env_.release();
        } else {
            env_.reset();
        }
    }

protected:
    void SetUp() override
    {
        ASSERT_FALSE(failedSynchronization_);
    }

    void Check(const Pattern &pattern)
    {
        std::vector<uint32_t> coreCounts = {1U, 2U, runtimeCores_};
        std::sort(coreCounts.begin(), coreCounts.end());
        coreCounts.erase(std::unique(coreCounts.begin(), coreCounts.end()), coreCounts.end());
        for (const uint32_t cores : coreCounts) {
            SCOPED_TRACE(testing::Message() << "cores=" << cores << " nnz=" << pattern.tiling.nnzUnits);
            auto device = std::make_unique<DevicePattern>();
            ASSERT_NO_FATAL_FAILURE(Prepare(pattern, cores, *device));
            Launch(*device, pattern, cores, env_->stream());
            const auto status = aclrtSynchronizeStream(env_->stream());
            if (status != ACL_SUCCESS) {
                failedSynchronization_ = true;
                // Failed synchronization does not prove that Device writes
                // stopped. Keep this allocation alive for fault investigation.
                (void)device.release();
            }
            ASSERT_EQ(status, ACL_SUCCESS);
            ASSERT_NO_FATAL_FAILURE(CheckResult(pattern, cores, *device));
        }
    }

    void Measure(const Pattern &pattern, uint32_t cores)
    {
        constexpr int kWarmup = 2;
        constexpr int kLaunchesPerRound = 8;
        constexpr int kRounds = 3;
        auto device = std::make_unique<DevicePattern>();
        ASSERT_NO_FATAL_FAILURE(Prepare(pattern, cores, *device));
        auto events = std::make_unique<TimingEvents>();
        ASSERT_EQ(aclrtCreateEvent(&events->start), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateEvent(&events->stop), ACL_SUCCESS);
        for (int warmup = 0; warmup < kWarmup; ++warmup) {
            Launch(*device, pattern, cores, env_->stream());
        }
        auto status = aclrtSynchronizeStream(env_->stream());
        if (status != ACL_SUCCESS) {
            failedSynchronization_ = true;
            (void)device.release();
            (void)events.release();
        }
        ASSERT_EQ(status, ACL_SUCCESS);
        std::vector<double> samples;
        for (int round = 0; round < kRounds; ++round) {
            ASSERT_EQ(aclrtRecordEvent(events->start, env_->stream()), ACL_SUCCESS);
            for (int launch = 0; launch < kLaunchesPerRound; ++launch) {
                Launch(*device, pattern, cores, env_->stream());
            }
            const auto recordStatus = aclrtRecordEvent(events->stop, env_->stream());
            status = aclrtSynchronizeStream(env_->stream());
            if (status != ACL_SUCCESS) {
                failedSynchronization_ = true;
                (void)device.release();
                (void)events.release();
            }
            ASSERT_EQ(status, ACL_SUCCESS);
            ASSERT_EQ(recordStatus, ACL_SUCCESS);
            float milliseconds = 0.0F;
            ASSERT_EQ(aclrtEventElapsedTime(&milliseconds, events->start, events->stop), ACL_SUCCESS);
            ASSERT_TRUE(std::isfinite(milliseconds));
            ASSERT_GE(milliseconds, 0.0F);
            samples.push_back(milliseconds * 1000.0 / kLaunchesPerRound);
        }
        ASSERT_NO_FATAL_FAILURE(CheckResult(pattern, cores, *device));
        PrintTimingSamples(pattern, cores, samples, kWarmup, kLaunchesPerRound, kRounds);
    }

    inline static uint32_t runtimeCores_ = 0;
    inline static bool failedSynchronization_ = false;

private:
    inline static std::unique_ptr<AclEnvScope> env_;
};

TEST_F(SddmmPatternNpuTest, CsrSortedCyclic64Stride17BothIndexBases)
{
    for (const int32_t base : {0, 1}) {
        auto pattern = MakeCyclic(13, 97, base, 17);
        ASSERT_EQ(SerialOracle(pattern).flags, sddmm_arch35::kSddmmAllPatternFlags);
        ASSERT_NO_FATAL_FAILURE(Check(pattern));
    }
}

TEST_F(SddmmPatternNpuTest, CsrWeakerPatternAndUnevenBadColumnTail)
{
    auto pattern = MakeCyclic(7, 97, 0, 9);
    ASSERT_EQ(SerialOracle(pattern).flags, SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern.columns.back() = pattern.columns[pattern.columns.size() - 2];
    ASSERT_EQ(SerialOracle(pattern).flags, SDDMM_PATTERN_FLAG_NONE);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern = MakeCsr({0, 1, 65, 2, 73, 0, 3}, 97, 1);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern.columns.back() = 98;
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_COLUMN);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
}

TEST_F(SddmmPatternNpuTest, CsrEmptyNnzNonmonotonicOffsetsAndWrongEndpoints)
{
    for (const int32_t base : {0, 1}) {
        ASSERT_NO_FATAL_FAILURE(Check(MakeCsr({0, 0, 0, 0, 0, 0, 0}, 97, base)));
        auto pattern = MakeCsr({2, 2, 2, 2, 2}, 97, base);
        pattern.offsets[3] = pattern.offsets[2] - 1;
        ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_NOT_MONOTONIC);
        ASSERT_NO_FATAL_FAILURE(Check(pattern));
        pattern = MakeCsr({2, 2, 2, 2, 2}, 97, base);
        --pattern.offsets.back();
        ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_ENDPOINT);
        ASSERT_NO_FATAL_FAILURE(Check(pattern));
    }
}

TEST_F(SddmmPatternNpuTest, BsrBroadcastOffsetsIndependentColumnsLastBatchFailure)
{
    auto pattern = MakeBsr(false, true, 0);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern.columns[2 * pattern.tiling.columnsBatchStride + 12] = 97;
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_COLUMN);
    ASSERT_EQ(SerialOracle(pattern).firstInvalidBatch, 2);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
}

TEST_F(SddmmPatternNpuTest, BsrIndependentOffsetsBroadcastColumnsLastBatchEndpoint)
{
    auto pattern = MakeBsr(true, false, 1);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    --pattern.offsets[2 * pattern.tiling.offsetsBatchStride + 5];
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_ENDPOINT);
    ASSERT_EQ(SerialOracle(pattern).firstInvalidBatch, 2);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
}

TEST_F(SddmmPatternNpuTest, BsrIndependentPaddedStridesLastBatchNonmonotonicOffsets)
{
    auto pattern = MakeBsr(true, true, 1);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    const int64_t lastBatch = 2 * pattern.tiling.offsetsBatchStride;
    pattern.offsets[lastBatch + 2] = pattern.offsets[lastBatch + 1] - 1;
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_NOT_MONOTONIC);
    ASSERT_EQ(SerialOracle(pattern).firstInvalidBatch, 2);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
}

TEST_F(SddmmPatternNpuTest, BsrEarliestBatchWinsBeforeErrorClassPriority)
{
    auto pattern = MakeBsr(true, true, 0);
    pattern.columns[12] = 97;
    --pattern.offsets[pattern.tiling.offsetsBatchStride + 5];
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_COLUMN);
    ASSERT_EQ(SerialOracle(pattern).firstInvalidBatch, 0);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern.columns[12] = 1;
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_BAD_ENDPOINT);
    ASSERT_EQ(SerialOracle(pattern).firstInvalidBatch, 1);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
    pattern.columns[12] = 97;
    pattern.offsets[2] = pattern.offsets[1] - 1;
    ASSERT_EQ(SerialOracle(pattern).status, SDDMM_PATTERN_NOT_MONOTONIC);
    ASSERT_NO_FATAL_FAILURE(Check(pattern));
}

TEST_F(SddmmPatternNpuTest, SameCandidateKernelEventOneCoreAndRuntimeCores)
{
    constexpr int32_t kRows = 2048;
    for (const int32_t nnz : {1000000, 4000000}) {
        std::vector<int32_t> degrees(kRows, nnz / kRows);
        for (int32_t row = 0; row < nnz % kRows; ++row) {
            ++degrees[row];
        }
        const auto pattern = MakeCsr(degrees, 257, 0);
        const size_t inputBytes = (pattern.offsets.size() + pattern.columns.size()) * sizeof(int32_t);
        ASSERT_LE(inputBytes, 16U * 1024U * 1024U);
        ASSERT_NO_FATAL_FAILURE(Measure(pattern, 1U));
        ASSERT_NO_FATAL_FAILURE(Measure(pattern, runtimeCores_));
    }
}

} // namespace
} // namespace sddmm_pattern
} // namespace sparse_test
