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

// All SDDMM cases remain registered in this existing test entry point.
#include "sddmm_test_support.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

#include "tiling/platform/platform_ascendc.h"

namespace sparse_test {
namespace sddmm_extended {

TEST(SddmmCubeResourceTest, ComplexBsrL0CapacityBoundaries)
{
    CheckComplexBsrL0CapacityBoundaries();
}

TEST(SddmmCubeResourceTest, ComplexBsrGroupedL0Requirement)
{
    CheckComplexBsrGroupedL0Requirement();
}

TEST(SddmmCubeResourceTest, ComplexBsrL0RejectsUnsupportedTiles)
{
    CheckComplexBsrL0RejectsUnsupportedTiles();
}

class SddmmExtendedNpuTest : public testing::Test {
public:
    static void SetUpTestSuite()
    {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite()
    {
        env_.reset();
    }

protected:
    aclrtStream stream() const { return env_->stream(); }

private:
    inline static std::unique_ptr<AclEnvScope> env_;
};

TEST_F(SddmmExtendedNpuTest, WarmPackedYInvalidatesAliasedWorkspaceOwners)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceOwners(stream(), false));
}

TEST_F(SddmmExtendedNpuTest, WarmCsrCubeInvalidatesAliasedWorkspaceOwners)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceOwners(stream(), true));
}

#ifdef SDDMM_TEST_WITH_SPMM
TEST_F(SddmmExtendedNpuTest, WarmPackedYSpmmSharedWorkspace)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceSpmm(stream(), false, true));
}

TEST_F(SddmmExtendedNpuTest, WarmPackedYSpmmSeparateWorkspace)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceSpmm(stream(), false, false));
}

TEST_F(SddmmExtendedNpuTest, WarmCsrCubeSpmmSharedWorkspace)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceSpmm(stream(), true, true));
}

TEST_F(SddmmExtendedNpuTest, WarmCsrCubeSpmmSeparateWorkspace)
{
    ASSERT_NO_FATAL_FAILURE(CheckWarmWorkspaceSpmm(stream(), true, false));
}

#endif

TEST_F(SddmmExtendedNpuTest, ComplexBetaOneInfinityGeneric)
{
    for (const bool deviceScalars : {false, true}) {
        SCOPED_TRACE(deviceScalars);
        ASSERT_NO_FATAL_FAILURE(CheckCsrComplexBetaOneInfinity(stream(), deviceScalars, 17));
        ASSERT_NO_FATAL_FAILURE(CheckBsrComplexBetaOneInfinity(
            stream(), 4, 32, ACL_SPARSE_ORDER_ROW, deviceScalars));
    }
}

TEST_F(SddmmExtendedNpuTest, ComplexBetaOneInfinityCsrRow)
{
    ASSERT_NO_FATAL_FAILURE(CheckCsrComplexBetaOneInfinity(stream(), false, 128));
}

TEST_F(SddmmExtendedNpuTest, ComplexBetaOneInfinityBsrCube)
{
    for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
        SCOPED_TRACE(order);
        ASSERT_NO_FATAL_FAILURE(CheckBsrComplexBetaOneInfinity(stream(), 16, 128, order, false));
    }
}

TEST_F(SddmmExtendedNpuTest, CsrBaseOneComplex64)
{
    const auto result = RunCsrBaseOneComplex(stream());
    ASSERT_TRUE(StagesSucceeded(result.stages));
    EXPECT_TRUE(VerifyComplexRun(result, "CSR_BASE1_COMPLEX64"));
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64CubeBlock16Block32Block64)
{
    std::vector<BsrComplexCase> cases;
    AddComplexCubeBlock16Cases(cases);
    AddComplexCubeBlock32Cases(cases);
    AddComplexCubeBlock64Cases(cases);
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrComplexCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        ASSERT_GT(result.stages.workspaceSize,
                  static_cast<size_t>(kSddmmWorkspaceBytes));
        EXPECT_LE(result.stages.workspaceSize - kSddmmWorkspaceBytes,
                  static_cast<size_t>(
                      sddmm_arch35::kSddmmComplexBsrCubeWorkspaceLimit));
        EXPECT_TRUE(VerifyComplexRun(result, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64ColumnIdentityMultipleTilesAndTail)
{
    for (const int64_t reduction : {64, 128}) {
        SCOPED_TRACE(reduction);
        auto testCase = MakeComplexColumnIdentityCase(reduction);
        ComplexRunResult rowResult;
        {
            SCOPED_TRACE("ROW");
            testCase.name = "BSR_COMPLEX64_B16_ROW_IDENTITY_MULTIPLE_TILES_TAIL";
            rowResult = RunBsrComplexCase(stream(), testCase);
            ASSERT_TRUE(StagesSucceeded(rowResult.stages));
            ASSERT_TRUE(VerifyComplexRun(rowResult, testCase.name));
        }
        ComplexRunResult columnResult;
        {
            SCOPED_TRACE("COL");
            testCase.name = "BSR_COMPLEX64_B16_COL_IDENTITY_MULTIPLE_TILES_TAIL";
            testCase.blockOrder = ACL_SPARSE_ORDER_COL;
            columnResult = RunBsrComplexCase(stream(), testCase);
            ASSERT_TRUE(StagesSucceeded(columnResult.stages));
            ASSERT_GT(columnResult.stages.workspaceSize,
                static_cast<size_t>(kSddmmWorkspaceBytes));
            ASSERT_TRUE(VerifyComplexRun(columnResult, testCase.name));
        }
        ASSERT_EQ(rowResult.actualParts.size(), columnResult.actualParts.size());
        for (size_t block = 0; block < testCase.zeroBasedColumns.size(); ++block) {
            for (size_t physical = 0; physical < 256U; ++physical) {
                const size_t logical = (physical % 16U) * 16U + physical / 16U;
                ASSERT_EQ(std::memcmp(
                    &columnResult.actualParts[(block * 256U + physical) * 2U],
                    &rowResult.actualParts[(block * 256U + logical) * 2U],
                    2U * sizeof(float)), 0) << "block=" << block << " physical=" << physical;
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64FourBlockGroupsPreservePhysicalOutputs)
{
    for (int variant = 0; variant < 4; ++variant) {
        auto testCase = MakeComplexFourGroupCase(variant);
        const bool identity = variant < 2;
        SCOPED_TRACE(testCase.name);
        const auto grouped = RunBsrComplexCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(grouped.stages));
        ASSERT_TRUE(VerifyComplexRun(grouped, testCase.name));
        const uint64_t expectedWorkspace = kSddmmWorkspaceBytes +
            static_cast<uint64_t>(testCase.blockRows * 16 * 128) * 16U +
            static_cast<uint64_t>(testCase.blockCols * 16 * 128) * 8U +
            testCase.zeroBasedColumns.size() * 16U * 16U * 8U;
        ASSERT_EQ(grouped.stages.workspaceSize, expectedWorkspace);
        if (!identity) {
            continue;
        }

        BsrComplexCase fallback = testCase;
        fallback.name += "_SINGLE_BLOCK_REFERENCE";
        const auto originalSlots = BuildDescendingComplexReference(testCase, fallback);
        const auto reference = RunBsrComplexCase(stream(), fallback);
        ASSERT_TRUE(StagesSucceeded(reference.stages));
        ASSERT_TRUE(VerifyComplexRun(reference, fallback.name));
        ASSERT_EQ(reference.actualParts.size(), grouped.actualParts.size());
        constexpr size_t kPartsPerBlock = 16U * 16U * 2U;
        for (size_t slot = 0; slot < originalSlots.size(); ++slot) {
            ASSERT_EQ(std::memcmp(&reference.actualParts[slot * kPartsPerBlock],
                &grouped.actualParts[originalSlots[slot] * kPartsPerBlock],
                kPartsPerBlock * sizeof(float)), 0) << "reference slot=" << slot;
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64SparseYPackBoundariesAndDuplicateColumns)
{
    const SparsePackVariant variants[] = {
        {1, 2, 64, 128, true}, {8, 16, 64, 128, true},
        {20, 40, 64, 128, true}, {64, 128, 64, 128, true},
        {65, 130, 64, 128, false}, {8, 15, 64, 128, false},
        {8, 16, 32, 128, false}, {8, 16, 64, 64, false}};
    int ordinal = 0;
    for (const auto &variant : variants) {
        auto testCase = MakeSparsePackCase(variant, ordinal);
        const bool identity = ordinal == 0 || ordinal == 3 || ordinal == 4 || ordinal == 7;
        SddmmTilingData tiling{};
        tiling.m = testCase.blockRows * testCase.blockSize;
        tiling.n = testCase.blockCols * testCase.blockSize;
        tiling.k = testCase.k;
        tiling.blockDim = testCase.blockSize;
        tiling.nnzUnits = variant.blocks;
        tiling.outputsPerBatch = variant.blocks * testCase.blockSize * testCase.blockSize;
        SCOPED_TRACE(testCase.name);
        ASSERT_EQ(sddmm_arch35::SddmmComplexBsrCubeUseSparseYPack(tiling), variant.sparsePack);
        const auto result = RunBsrComplexCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        ASSERT_EQ(result.stages.workspaceSize, kSddmmWorkspaceBytes +
            sddmm_arch35::SddmmComplexBsrCubeWorkspaceBytes(tiling));
        ASSERT_TRUE(VerifyComplexRun(result, testCase.name));
        if (variant.sparsePack && identity) {
            // Appending physical blocks disables sparse packing while leaving
            // every original X/Y sample and identity output unchanged.
            BsrComplexCase densePack = testCase;
            densePack.name += "_DENSE_PACK_REFERENCE";
            densePack.zeroBasedColumns.resize(65, testCase.zeroBasedColumns.back());
            densePack.zeroBasedOffsets.back() = 65;
            const auto reference = RunBsrComplexCase(stream(), densePack);
            ASSERT_TRUE(StagesSucceeded(reference.stages));
            ASSERT_TRUE(VerifyComplexRun(reference, densePack.name));
            ASSERT_GE(reference.actualParts.size(), result.actualParts.size());
            ASSERT_EQ(std::memcmp(result.actualParts.data(), reference.actualParts.data(),
                result.actualParts.size() * sizeof(float)), 0);
        }
        ++ordinal;
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64GenericBlock4Block8AndColumnMajor)
{
    BsrComplexCase block4ColumnMajor;
    block4ColumnMajor.name = "BSR_COMPLEX64_GENERIC_B4_BASE0_COL";
    block4ColumnMajor.blockSize = 4;
    block4ColumnMajor.k = 7;
    block4ColumnMajor.blockOrder = ACL_SPARSE_ORDER_COL;

    BsrComplexCase block8BaseOne;
    block8BaseOne.name = "BSR_COMPLEX64_GENERIC_B8_BASE1_ROW";
    block8BaseOne.blockSize = 8;
    block8BaseOne.k = 11;
    block8BaseOne.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    block8BaseOne.alpha = {-0.625F, 0.375F};
    block8BaseOne.beta = {0.25F, -0.75F};

    const BsrComplexCase cases[] = {block4ColumnMajor, block8BaseOne};
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrComplexCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        // b4/b8 are intentionally outside the b16/b32/b64 complex Cube
        // predicate, so only the generic preprocess prefix is required.
        EXPECT_EQ(result.stages.workspaceSize,
                  static_cast<size_t>(kSddmmWorkspaceBytes));
        EXPECT_TRUE(VerifyComplexRun(result, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64Block4K32MatchesGeneric)
{
    for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
        for (const auto base : {ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE}) {
            for (int scalarMode = 0; scalarMode < 3; ++scalarMode) {
                BsrComplexCase testCase;
                testCase.name = "BSR_COMPLEX64_B4_K32_GENERIC_COMPARISON";
                testCase.blockSize = 4;
                testCase.blockRows = 100;
                testCase.blockCols = 37;
                testCase.k = 32;
                testCase.blockOrder = order;
                testCase.indexBase = base;
                testCase.randomValues = true;
                testCase.guardedRepeat = true;
                testCase.deviceScalars = scalarMode == 1 ||
                    base == ACL_SPARSE_INDEX_BASE_ONE;
                if (scalarMode == 0) {
                    testCase.alpha = {1.0F, 0.0F};
                    testCase.beta = {0.0F, 0.0F};
                } else if (scalarMode == 2) {
                    testCase.alpha = {0.0F, 0.0F};
                    testCase.specialDenseValues = true;
                }
                testCase.zeroBasedOffsets = {0};
                testCase.zeroBasedColumns.clear();
                // Empty and unequal rows exercise binary-search ownership;
                // 28800 scalar outputs cross the target grid stride with a tail.
                constexpr int32_t rowLengths[] = {0, 1, 35, 36};
                for (int32_t row = 0; row < testCase.blockRows; ++row) {
                    for (int32_t column = 0; column < rowLengths[row % 4]; ++column) {
                        testCase.zeroBasedColumns.push_back((column + row * 5) % 37);
                    }
                    testCase.zeroBasedOffsets.push_back(
                        static_cast<int32_t>(testCase.zeroBasedColumns.size()));
                }
                SCOPED_TRACE(::testing::Message() << "order=" << order << " base=" << base
                    << " scalarMode=" << scalarMode << " deviceScalars=" << testCase.deviceScalars);
                const auto specialized = RunBsrComplexCase(stream(), testCase);
                ASSERT_TRUE(StagesSucceeded(specialized.stages));
                ASSERT_TRUE(VerifyComplexRun(specialized, testCase.name));
                // The same logical Y in COL layout deliberately exercises
                // the existing generic kernel as an independent bitwise reference.
                testCase.columnMajorY = true;
                const auto generic = RunBsrComplexCase(stream(), testCase);
                ASSERT_TRUE(StagesSucceeded(generic.stages));
                ASSERT_TRUE(VerifyComplexRun(generic, testCase.name + "_GENERIC"));
                ASSERT_EQ(specialized.actualParts.size(), generic.actualParts.size());
                ASSERT_EQ(std::memcmp(specialized.actualParts.data(), generic.actualParts.data(),
                    specialized.actualParts.size() * sizeof(float)), 0);
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplex64EightByteAlignedInputFallsBack)
{
    BsrComplexCase testCase;
    testCase.name = "BSR_COMPLEX64_B16_OFFSET8_GENERIC";
    testCase.offsetDensePointersByOneElement = true;
    const auto result = RunBsrComplexCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.stages));
    EXPECT_EQ(result.stages.workspaceSize,
              static_cast<size_t>(kSddmmWorkspaceBytes));
    EXPECT_TRUE(VerifyComplexRun(result, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, BsrSupportMatrix)
{
    BsrRealCase bf16Output;
    bf16Output.name = "BSR_B2_BASE0_ROW_BF16_BF16";
    bf16Output.blockSize = 2;
    bf16Output.inputType = ACL_BF16;
    bf16Output.outputType = ACL_BF16;

    BsrRealCase bf16Fp32;
    bf16Fp32.name = "BSR_B16_BASE1_COL_BF16_FP32";
    bf16Fp32.blockSize = 16;
    bf16Fp32.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    bf16Fp32.blockOrder = ACL_SPARSE_ORDER_COL;
    bf16Fp32.inputType = ACL_BF16;
    bf16Fp32.outputType = ACL_FLOAT;

    BsrRealCase fp16Fp32;
    fp16Fp32.name = "BSR_B128_BASE0_ROW_FP16_FP32";
    fp16Fp32.blockSize = 128;
    fp16Fp32.inputType = ACL_FLOAT16;
    fp16Fp32.outputType = ACL_FLOAT;

    const BsrRealCase cases[] = {bf16Output, bf16Fp32, fp16Fp32};
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_TRUE(VerifyRealRun(
            result, testCase.outputType, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrStridedBatchBroadcastCombinations)
{
    const int batchCounts[][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
    for (const auto &batchCount : batchCounts) {
        BsrRealCase testCase;
        testCase.name = "BSR_BATCH_X" + std::to_string(batchCount[0]) +
                        "_Y" + std::to_string(batchCount[1]);
        testCase.blockRows = 2;
        testCase.blockCols = 2;
        testCase.zeroBasedOffsets = {0, 2, 3};
        testCase.zeroBasedColumns = {0, 1, 1};
        testCase.outputBatchCount = 2;
        testCase.xBatchCount = batchCount[0];
        testCase.yBatchCount = batchCount[1];
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_TRUE(VerifyRealRun(
            result, testCase.outputType, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrCubeK128BatchBaseOneColumnBf16)
{
    BsrRealCase testCase;
    testCase.name = "BSR_CUBE_B16_K128_BATCH2_BASE1_COL_BF16";
    testCase.blockRows = 2;
    testCase.blockCols = 3;
    testCase.blockSize = 16;
    testCase.k = 128;
    testCase.zeroBasedOffsets = {0, 2, 3};
    testCase.zeroBasedColumns = {0, 2, 1};
    testCase.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    testCase.blockOrder = ACL_SPARSE_ORDER_COL;
    testCase.inputType = ACL_BF16;
    testCase.outputType = ACL_BF16;
    testCase.outputBatchCount = 2;
    testCase.xBatchCount = 2;
    testCase.yBatchCount = 1;
    testCase.alpha = -0.625F;
    testCase.beta = 0.375F;
    const auto result = RunBsrRealCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.stages));
    constexpr size_t expectedScratch = 2U * 3U * 16U * 16U * sizeof(float);
    EXPECT_EQ(result.stages.workspaceSize,
              kSddmmWorkspaceBytes + expectedScratch);
    EXPECT_TRUE(VerifyRealRun(
        result, testCase.outputType, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, BsrCubeBlock16Fp32DirectOutputBaseZeroAndOne)
{
    for (const auto indexBase : {
             ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE}) {
        BsrRealCase testCase;
        testCase.name = indexBase == ACL_SPARSE_INDEX_BASE_ZERO ?
            "BSR_CUBE_B16_K128_DIRECT_FP32_BASE0" :
            "BSR_CUBE_B16_K128_DIRECT_FP32_BASE1";
        testCase.blockRows = 2;
        testCase.blockCols = 3;
        testCase.blockSize = 16;
        testCase.k = 128;
        testCase.zeroBasedOffsets = {0, 2, 3};
        testCase.zeroBasedColumns = {0, 2, 1};
        testCase.indexBase = indexBase;
        testCase.blockOrder = ACL_SPARSE_ORDER_ROW;
        testCase.inputType = ACL_FLOAT;
        testCase.outputType = ACL_FLOAT;
        testCase.alpha = 1.0F;
        testCase.beta = 0.0F;

        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        constexpr size_t expectedScratch =
            3U * 16U * 16U * sizeof(float);
        EXPECT_EQ(result.stages.workspaceSize,
                  kSddmmWorkspaceBytes + expectedScratch);
        ASSERT_TRUE(result.valuesPointer64Aligned);
        EXPECT_TRUE(VerifyRealRun(
            result, testCase.outputType, testCase.name));
        EXPECT_EQ(result.actual, result.golden);
    }
}

TEST_F(SddmmExtendedNpuTest, BsrCubeK128SameTypeBlock32And64)
{
    BsrRealCase fp16Block32;
    fp16Block32.name = "BSR_CUBE_B32_K128_BASE0_COL_FP16";
    fp16Block32.blockRows = 2;
    fp16Block32.blockCols = 3;
    fp16Block32.blockSize = 32;
    fp16Block32.k = 128;
    fp16Block32.zeroBasedOffsets = {0, 2, 3};
    fp16Block32.zeroBasedColumns = {2, 0, 1};
    fp16Block32.indexBase = ACL_SPARSE_INDEX_BASE_ZERO;
    fp16Block32.blockOrder = ACL_SPARSE_ORDER_COL;
    fp16Block32.inputType = ACL_FLOAT16;
    fp16Block32.outputType = ACL_FLOAT16;
    fp16Block32.alpha = 0.75F;
    fp16Block32.beta = -0.5F;

    BsrRealCase fp32Block64;
    fp32Block64.name = "BSR_CUBE_B64_K128_BASE1_ROW_FP32";
    fp32Block64.blockRows = 2;
    fp32Block64.blockCols = 2;
    fp32Block64.blockSize = 64;
    fp32Block64.k = 128;
    fp32Block64.zeroBasedOffsets = {0, 1, 3};
    fp32Block64.zeroBasedColumns = {1, 0, 1};
    fp32Block64.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    fp32Block64.blockOrder = ACL_SPARSE_ORDER_ROW;
    fp32Block64.inputType = ACL_FLOAT;
    fp32Block64.outputType = ACL_FLOAT;
    fp32Block64.alpha = -1.25F;
    fp32Block64.beta = 0.375F;

    struct CubeCase {
        const BsrRealCase *testCase;
        size_t expectedWorkspaceSize;
    };
    const CubeCase cases[] = {
        {&fp16Block32,
            kSddmmWorkspaceBytes + 3U * 32U * 32U * sizeof(float)},
        {&fp32Block64,
            kSddmmWorkspaceBytes + 3U * 64U * 64U * sizeof(float)},
    };
    for (const auto &entry : cases) {
        const auto &testCase = *entry.testCase;
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_EQ(result.stages.workspaceSize, entry.expectedWorkspaceSize);
        EXPECT_TRUE(VerifyRealRun(
            result, testCase.outputType, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, RealAlphaZeroSuppressesNanInf)
{
    CsrCubeRealCase csrCase;
    csrCase.name = "CSR_CUBE_ELIGIBLE_ALPHA0_NAN_INF";
    csrCase.dataType = ACL_FLOAT;
    csrCase.alpha = 0.0F;
    csrCase.beta = -0.375F;
    csrCase.specialDenseValues = true;
    const auto csrResult = RunCsrCubeRealCase(stream(), csrCase);
    ASSERT_TRUE(StagesSucceeded(csrResult.run.stages));
    EXPECT_NE(csrResult.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    EXPECT_TRUE(VerifyRealRun(csrResult.run, ACL_FLOAT, csrCase.name));

    for (const bool deviceScalars : {false, true}) {
        BsrRealCase bsrCase;
        bsrCase.name = deviceScalars ?
            "BSR_CUBE_ALPHA0_NAN_INF_DEVICE" :
            "BSR_CUBE_ALPHA0_NAN_INF_HOST";
        bsrCase.blockRows = 1;
        bsrCase.blockCols = 2;
        bsrCase.blockSize = 16;
        bsrCase.k = 128;
        bsrCase.alpha = 0.0F;
        bsrCase.beta = 0.625F;
        bsrCase.deviceScalars = deviceScalars;
        bsrCase.specialDenseValues = true;
        SCOPED_TRACE(bsrCase.name);
        const auto bsrResult = RunBsrRealCase(stream(), bsrCase);
        ASSERT_TRUE(StagesSucceeded(bsrResult.stages));
        if (deviceScalars) {
            // The real BSR Cube predicate is Host-pointer-only. Device scalars
            // use the generic kernel so alpha=0 is resolved before X/Y loads.
            EXPECT_EQ(bsrResult.stages.workspaceSize,
                      static_cast<size_t>(kSddmmWorkspaceBytes));
        } else {
            EXPECT_GT(bsrResult.stages.workspaceSize,
                      static_cast<size_t>(kSddmmWorkspaceBytes));
        }
        EXPECT_TRUE(VerifyRealRun(
            bsrResult, bsrCase.outputType, bsrCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, DevicePointerModeScalars)
{
    BsrRealCase testCase;
    testCase.name = "BSR_CUBE_ELIGIBLE_DEVICE_SCALARS_GENERIC_FALLBACK";
    testCase.blockRows = 1;
    testCase.blockCols = 2;
    testCase.blockSize = 16;
    testCase.k = 128;
    testCase.deviceScalars = true;
    testCase.alpha = -0.625F;
    testCase.beta = 0.375F;
    const auto result = RunBsrRealCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.stages));
    EXPECT_EQ(result.stages.workspaceSize,
              static_cast<size_t>(kSddmmWorkspaceBytes));
    EXPECT_TRUE(VerifyRealRun(
        result, testCase.outputType, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, DeviceIdentityNoOpPreservesExactValueBits)
{
    struct TypePair { aclDataType input; aclDataType output; };
    const TypePair types[] = {{ACL_FLOAT, ACL_FLOAT}, {ACL_FLOAT16, ACL_FLOAT16},
        {ACL_FLOAT16, ACL_FLOAT}, {ACL_BF16, ACL_BF16},
        {ACL_BF16, ACL_FLOAT}, {ACL_COMPLEX64, ACL_COMPLEX64}};
    size_t cases = 0;
    for (const bool bsr : {false, true}) {
        for (const auto &type : types) {
            if (!bsr && type.input == ACL_BF16) {
                continue;
            }
            for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
                if (!bsr && order == ACL_SPARSE_ORDER_COL) {
                    continue;
                }
                for (const auto base : {ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE}) {
                    SCOPED_TRACE("device identity case=" + std::to_string(cases++));
                    const DeviceIdentityCase testCase{bsr, type.input, type.output, order, base};
                    ASSERT_NO_FATAL_FAILURE(RunDeviceIdentityNoOpCase(stream(), testCase));
                }
            }
        }
    }
    EXPECT_EQ(cases, 32U);
}

TEST_F(SddmmExtendedNpuTest, CsrPreparedIdentityNoOpPreservesValues)
{
    const CsrCubeRealCase testCase = {
        "CSR_PREPARED_IDENTITY_NOOP_BASE1", ACL_FLOAT,
        ACL_SPARSE_INDEX_BASE_ONE, 64, false, 0.0F, 1.0F,
        false, 1, 96};
    const auto result = RunCsrCubeRealCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.run.stages));
    EXPECT_EQ(result.run.stages.workspaceSize,
        static_cast<size_t>(kSddmmWorkspaceBytes));
    EXPECT_NE(result.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
    EXPECT_NE(result.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    EXPECT_EQ(result.preparedLogicalFirstColumn,
        testCase.logicalFirstColumn);
    EXPECT_TRUE(VerifyRealRun(
        result.run, ACL_FLOAT, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, CsrPatternSetterInvalidatesPreparedState)
{
    const auto result = RunPreparedInvalidationCase(stream());
    ASSERT_EQ(result.stages.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(result.stages.workspaceSizeAligned);
    ASSERT_TRUE(result.stages.workspacePointerAligned);
    ASSERT_EQ(result.stages.preprocessRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.setterRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.stages.executeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.stages.synchronizeRet, ACL_SUCCESS);
    EXPECT_TRUE(result.preparedBefore);
    EXPECT_FALSE(result.preparedAfterSetter);
    EXPECT_TRUE(result.workspaceClearedAfterSetter);
    EXPECT_GT(result.generationAfterSetter, result.generationBefore);
    EXPECT_TRUE(result.preparedAfterExecute);
    EXPECT_GT(result.epochAfter, result.epochBefore);
    EXPECT_TRUE(result.rebuiltForNewOffsets);
    EXPECT_TRUE(VerifyRealRun(
        RealRunResult{result.stages, result.actual, result.golden},
        ACL_FLOAT, "CSR_PATTERN_REBUILD"));
}

TEST_F(SddmmExtendedNpuTest, SddmmInvalidatesOnlyAliasedActiveBuffer)
{
    const auto result = RunActiveBufferIsolationCase(stream());
    ASSERT_EQ(result.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.distinctWorkspaceRet, ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(result.aliasedWorkspaceRet, ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_TRUE(result.distinctOwnerPreserved);
    EXPECT_TRUE(result.aliasedOwnerCleared);
    EXPECT_TRUE(result.distinctSpmvOwnerPreserved);
    EXPECT_TRUE(result.aliasedSpmvOwnerCleared);
    EXPECT_TRUE(result.sddmmPrepared);
}

TEST_F(SddmmExtendedNpuTest, CsrFp16ColumnMajorUniform64)
{
    const aclsparseIndexBase_t bases[] = {
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE};
    for (const auto base : bases) {
        const std::string name = base == ACL_SPARSE_INDEX_BASE_ZERO ?
            "CSR_FP16_COL_UNIFORM64_BASE0" : "CSR_FP16_COL_UNIFORM64_BASE1";
        SCOPED_TRACE(name);
        const auto result = RunCsrFp16ColumnMajorCase(stream(), base);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_EQ(result.stages.workspaceSize, kSddmmWorkspaceBytes);
        EXPECT_NE(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
        EXPECT_EQ(result.preparedLogicalFirstColumn,
            static_cast<int32_t>(20260912U % 96U));
        EXPECT_TRUE(VerifyRealRun(result, ACL_FLOAT16, name));
    }
}

TEST_F(SddmmExtendedNpuTest, CsrCubeStaticFastPathCorrectness)
{
    const CsrCubeRealCase cases[] = {
        {"CSR_CUBE_FP16_BASE0_NO_WRAP", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ZERO, 64, false, 0.75F, -0.25F},
        {"CSR_CUBE_FP16_BASE1_WRAP", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ONE, 464, false, -0.625F, 0.375F},
        {"CSR_CUBE_FP32_BASE0_WRAP", ACL_FLOAT,
            ACL_SPARSE_INDEX_BASE_ZERO, 464, false, 0.5F, -0.75F},
        {"CSR_CUBE_FP32_BASE1_NO_WRAP", ACL_FLOAT,
            ACL_SPARSE_INDEX_BASE_ONE, 64, false, -1.25F, 0.5F},
        {"CSR_CUBE_FP16_FP32_BASE0_NO_WRAP", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ZERO, 64, false, 0.625F, -0.375F,
            true},
        {"CSR_CUBE_FP16_FP32_BASE1_SCRATCH_DOMINANT_WRAP", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ONE, 464, false, -0.75F, 0.3125F,
            true, 128, 512},
        {"CSR_CUBE_BF16_FP32_BASE1_WRAP", ACL_BF16,
            ACL_SPARSE_INDEX_BASE_ONE, 464, false, -0.875F, 0.25F,
            true},
        {"CSR_CUBE_FP16_BASE0_IDENTITY_DEPHASE", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ZERO, 0, false, 1.0F, 0.0F,
            false, 2176, 2048},
    };
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunCsrCubeRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.run.stages));
        EXPECT_EQ(result.run.stages.workspaceSize,
            result.expectedStaticWorkspace);
        EXPECT_NE(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
        EXPECT_EQ(result.preparedLogicalFirstColumn,
            testCase.logicalFirstColumn);
        ASSERT_NO_FATAL_FAILURE(CheckCsrCubeNumerics(result, testCase,
            testCase.fp32Output && testCase.dataType == ACL_FLOAT16 &&
            testCase.m > sddmm_arch35::kSddmmCsrCubeRows));
    }
}

TEST_F(SddmmExtendedNpuTest, CsrCubeVectorGatherWrapSmoke)
{
    const CsrCubeRealCase testCase = {
        "CSR_CUBE_FP16_BASE1_VECTOR_GATHER_WRAP", ACL_FLOAT16,
        ACL_SPARSE_INDEX_BASE_ONE, 464, false, -0.625F, 0.375F,
        false, 16, 512};
    const auto result = RunCsrCubeRealCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.run.stages));
    EXPECT_EQ(result.run.stages.workspaceSize,
        result.expectedStaticWorkspace);
    EXPECT_NE(result.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    EXPECT_EQ(result.preparedLogicalFirstColumn,
        testCase.logicalFirstColumn);
    EXPECT_TRUE(VerifyRealRun(
        result.run, ACL_FLOAT16, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, CsrPackedVectorCyclic64OffsetsDerived)
{
    const CsrCubeRealCase cases[] = {
        {"CSR_PACKED_CYCLIC64_BASE0", ACL_FLOAT,
            ACL_SPARSE_INDEX_BASE_ZERO, 64, false, 0.625F, -0.375F,
            false, 4, 96, false,
            sddmm_arch35::kSddmmCsrCubeRowColumnStride},
        {"CSR_PACKED_CYCLIC64_BASE1", ACL_FLOAT,
            ACL_SPARSE_INDEX_BASE_ONE, 64, false, -0.75F, 0.3125F,
            false, 4, 96, false,
            sddmm_arch35::kSddmmCsrCubeRowColumnStride},
    };
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunCsrCubeRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.run.stages));
        EXPECT_NE(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
        EXPECT_NE(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
        EXPECT_EQ(result.preparedLogicalFirstColumn,
            testCase.logicalFirstColumn);
        EXPECT_EQ(result.run.stages.workspaceSize,
            static_cast<size_t>(kSddmmWorkspaceBytes) +
                result.expectedPackedYBytes);
        EXPECT_TRUE(VerifyRealRun(
            result.run, ACL_FLOAT, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, CsrWeakCyclic64PatternFallsBack)
{
    const CsrCubeRealCase testCase = {
        "CSR_WEAK_CYCLIC64_STRIDE19", ACL_FLOAT,
        ACL_SPARSE_INDEX_BASE_ONE, 64, false, -0.75F, 0.3125F,
        false, 4, 96, false, 19};
    const auto result = RunCsrCubeRealCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.run.stages));
    EXPECT_NE(result.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
    EXPECT_EQ(result.patternFlags &
        SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    EXPECT_TRUE(VerifyRealRun(
        result.run, ACL_FLOAT, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, CsrCubeSingleColumnPerturbationFallsBack)
{
    const CsrCubeRealCase cases[] = {
        {"CSR_CUBE_FP32_SINGLE_COLUMN_FALLBACK", ACL_FLOAT,
            ACL_SPARSE_INDEX_BASE_ONE, 464, true, 0.625F, -0.375F},
        {"CSR_CUBE_FP16_FP32_BASE0_SCRATCH_DOMINANT_FALLBACK", ACL_FLOAT16,
            ACL_SPARSE_INDEX_BASE_ZERO, 64, true, -0.75F, 0.3125F,
            true, 128, 512},
    };
    for (const auto &testCase : cases) {
        SCOPED_TRACE(testCase.name);
        const auto result = RunCsrCubeRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.run.stages));
        // A rejected Cube pattern only needs the generic packed-Y capacity,
        // including when the otherwise eligible Cube scratch would be larger.
        EXPECT_EQ(result.run.stages.workspaceSize,
            static_cast<size_t>(kSddmmWorkspaceBytes) + result.expectedPackedYBytes);
        EXPECT_EQ(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
        EXPECT_EQ(result.patternFlags &
            SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
        EXPECT_EQ(result.preparedLogicalFirstColumn, -1);
        const aclDataType outputType = testCase.fp32Output ? ACL_FLOAT : testCase.dataType;
        EXPECT_TRUE(VerifyRealRun(result.run, outputType, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrRealRandomFastPathsPreserveBoundsAndInputs)
{
    const aclDataType inputTypes[] = {ACL_FLOAT16, ACL_BF16, ACL_FLOAT};
    int ordinal = 0;
    for (const int64_t reduction : {32, 64, 128}) {
        for (const int64_t block : {16, 32, 64}) {
            for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
                BsrRealCase testCase;
                testCase.k = reduction;
                testCase.blockSize = block;
                testCase.blockRows = 3;
                testCase.blockCols = 4;
                testCase.blockOrder = order;
                testCase.indexBase = ordinal % 2 == 0 ?
                    ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
                testCase.inputType = inputTypes[ordinal % 3];
                testCase.outputType = ordinal % 2 == 0 ?
                    ACL_FLOAT : testCase.inputType;
                SetGuardedBsrPattern(ordinal % 3,
                    testCase.zeroBasedOffsets, testCase.zeroBasedColumns);
                testCase.alpha = -0.625F;
                testCase.beta = 0.375F;
                testCase.randomValues = true;
                testCase.guardedRepeat = true;
                testCase.name = "GUARDED_REAL_K" + std::to_string(reduction) +
                    "_B" + std::to_string(block) + "_CASE" +
                    std::to_string(ordinal++);
                SCOPED_TRACE(testCase.name);
                std::cout << testCase.name << " dtype=" << testCase.inputType
                          << " output=" << testCase.outputType
                          << " order=" << testCase.blockOrder
                          << " base=" << testCase.indexBase << std::endl;
                const auto result = RunBsrRealCase(stream(), testCase);
                ASSERT_TRUE(StagesSucceeded(result.stages));
                EXPECT_TRUE(VerifyRealRun(
                    result, testCase.outputType, testCase.name));
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrBlock8RandomGenericPathsPreserveBounds)
{
    int ordinal = 0;
    const int64_t reductions[] = {32, 64, 128};
    for (const aclDataType inputType : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
        for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
            BsrRealCase testCase;
            testCase.k = reductions[ordinal % 3];
            testCase.blockSize = 8;
            testCase.blockRows = 3;
            testCase.blockCols = 4;
            testCase.blockOrder = order;
            testCase.indexBase = ordinal % 2 == 0 ?
                ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
            testCase.inputType = inputType;
            testCase.outputType = ordinal % 2 == 0 ? ACL_FLOAT : inputType;
            SetGuardedBsrPattern(ordinal % 3,
                testCase.zeroBasedOffsets, testCase.zeroBasedColumns);
            testCase.alpha = -0.625F;
            testCase.beta = 0.375F;
            testCase.randomValues = true;
            testCase.guardedRepeat = true;
            testCase.name = "GUARDED_B8_REAL_CASE" + std::to_string(ordinal++);
            SCOPED_TRACE(testCase.name);
            std::cout << testCase.name << " dtype=" << inputType
                      << " output=" << testCase.outputType << " K=" << testCase.k
                      << " order=" << order << std::endl;
            const auto result = RunBsrRealCase(stream(), testCase);
            ASSERT_TRUE(StagesSucceeded(result.stages));
            EXPECT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrFp32ColumnIdentityTransposePreservesBoundsAndInputs)
{
    int ordinal = 0;
    for (const int64_t reduction : {32, 64, 128}) {
        for (const int64_t block : {16, 32, 64}) {
            BsrRealCase testCase;
            testCase.name = "GUARDED_FP32_COL_IDENTITY_K" + std::to_string(reduction) +
                "_B" + std::to_string(block);
            testCase.k = reduction;
            testCase.blockSize = block;
            testCase.blockRows = 3;
            testCase.blockCols = 4;
            testCase.blockOrder = ACL_SPARSE_ORDER_COL;
            testCase.indexBase = ordinal % 2 == 0 ?
                ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
            testCase.inputType = ACL_FLOAT;
            testCase.outputType = ACL_FLOAT;
            testCase.alpha = 1.0F;
            testCase.beta = 0.0F;
            testCase.randomValues = true;
            testCase.guardedRepeat = true;
            SetGuardedBsrPattern(ordinal++ % 3,
                testCase.zeroBasedOffsets, testCase.zeroBasedColumns);
            SCOPED_TRACE(testCase.name);
            ASSERT_EQ(testCase.blockOrder, ACL_SPARSE_ORDER_COL);
            ASSERT_EQ(testCase.alpha, 1.0F);
            ASSERT_EQ(testCase.beta, 0.0F);
            ASSERT_EQ(testCase.outputBatchCount, 1);
            ASSERT_FALSE(testCase.deviceScalars);
            std::cout << testCase.name << " base=" << testCase.indexBase << std::endl;
            const auto result = RunBsrRealCase(stream(), testCase);
            ASSERT_TRUE(StagesSucceeded(result.stages));
            // Canonical ROW dense descriptors and this nonempty b/K/dtype
            // combination select Cube. COL identity output excludes direct
            // ROW Fixpipe and selects the B32 vector Gather epilogue.
            const size_t scratchBytes = testCase.zeroBasedColumns.size() *
                static_cast<size_t>(block * block) * sizeof(float);
            EXPECT_EQ(result.stages.workspaceSize,
                static_cast<size_t>(kSddmmWorkspaceBytes) + scratchBytes);
            EXPECT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrBlock8VectorIdentityPreservesBoundsAndInputs)
{
    const aclDataType inputTypes[] = {ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT16};
    const aclDataType outputTypes[] = {ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT};
    int ordinal = 0;
    for (int type = 0; type < 3; ++type) {
        for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
            for (const auto base : {ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE}) {
                for (int pattern = 0; pattern < 2; ++pattern) {
                    const auto testCase = MakeBlock8IdentityCase(ordinal, inputTypes[type],
                        outputTypes[type], order, base, pattern);
                    const int64_t n = testCase.blockCols * testCase.blockSize;
                    const int64_t blockNnz = static_cast<int64_t>(testCase.zeroBasedColumns.size());
                    const int64_t outputs = blockNnz * testCase.blockSize * testCase.blockSize;
                    SCOPED_TRACE(testCase.name);
                    ASSERT_GE(n, 64);
                    ASSERT_EQ(testCase.k, 128);
                    ASSERT_EQ(testCase.alpha, 1.0F);
                    ASSERT_EQ(testCase.beta, 0.0F);
                    ASSERT_GE(outputs, 2 * n);
                    ASSERT_EQ(testCase.outputBatchCount, 1);
                    ASSERT_EQ(testCase.xBatchCount, 1);
                    ASSERT_EQ(testCase.yBatchCount, 1);
                    ASSERT_FALSE(testCase.deviceScalars);
                    std::cout << testCase.name << " input=" << testCase.inputType
                              << " output=" << testCase.outputType << " order=" << order
                              << " base=" << base << " pattern=" << pattern
                              << " blockNnz=" << blockNnz << std::endl;
                    const auto result = RunBsrRealCase(stream(), testCase);
                    ASSERT_TRUE(StagesSucceeded(result.stages));
                    const size_t elementBytes = testCase.inputType == ACL_FLOAT ?
                        sizeof(float) : sizeof(uint16_t);
                    // b8 cannot use Cube. An exact packed-Y reservation,
                    // canonical descriptors and identity Host scalars satisfy
                    // the separate vector predicate rather than the fallback.
                    EXPECT_EQ(result.stages.workspaceSize,
                        static_cast<size_t>(kSddmmWorkspaceBytes) +
                        static_cast<size_t>(testCase.k * n) * elementBytes);
                    EXPECT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
                }
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrBlock8Fp16SubnormalAndNanLoads)
{
    for (int mode = 0; mode < 3; ++mode) {
        BsrRealCase testCase;
        testCase.name = "GUARDED_B8_FP16_SPECIAL_MODE" + std::to_string(mode);
        testCase.blockSize = 8;
        testCase.k = 32;
        testCase.inputType = ACL_FLOAT16;
        testCase.outputType = mode == 0 ? ACL_FLOAT : ACL_FLOAT16;
        testCase.blockOrder = mode == 1 ? ACL_SPARSE_ORDER_COL : ACL_SPARSE_ORDER_ROW;
        testCase.alpha = mode == 2 ? 0.0F : 1.0F;
        testCase.beta = mode == 2 ? -0.5F : 0.0F;
        testCase.subnormalDenseValues = mode == 0;
        testCase.specialDenseValues = mode != 0;
        testCase.randomValues = true;
        testCase.guardedRepeat = true;
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
    }
}

TEST_F(SddmmExtendedNpuTest, BsrComplexRandomFastPathsPreserveBoundsAndInputs)
{
    int ordinal = 0;
    for (const int64_t reduction : {64, 128}) {
        for (const int64_t block : {16, 32, 64}) {
            for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
                BsrComplexCase testCase;
                testCase.k = reduction;
                testCase.blockSize = block;
                testCase.blockRows = 3;
                testCase.blockCols = 4;
                testCase.blockOrder = order;
                testCase.indexBase = ordinal % 2 == 0 ?
                    ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
                SetGuardedBsrPattern(ordinal % 3,
                    testCase.zeroBasedOffsets, testCase.zeroBasedColumns);
                testCase.alpha = {-0.625F, 0.375F};
                testCase.beta = {0.25F, -0.5F};
                testCase.randomValues = true;
                testCase.guardedRepeat = true;
                testCase.name = "GUARDED_COMPLEX_K" + std::to_string(reduction) +
                    "_B" + std::to_string(block) + "_CASE" +
                    std::to_string(ordinal++);
                SCOPED_TRACE(testCase.name);
                std::cout << testCase.name << " order=" << testCase.blockOrder
                          << " base=" << testCase.indexBase << std::endl;
                const auto result = RunBsrComplexCase(stream(), testCase);
                ASSERT_TRUE(StagesSucceeded(result.stages));
                EXPECT_TRUE(VerifyComplexRun(result, testCase.name));
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrGuardedAlphaZeroSuppressesNanInf)
{
    for (const aclDataType inputType : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
        BsrRealCase testCase;
        testCase.name = "GUARDED_ALPHA_ZERO_REAL_" +
            std::to_string(static_cast<int>(inputType));
        testCase.blockSize = 32;
        testCase.k = 64;
        testCase.inputType = inputType;
        testCase.outputType = ACL_FLOAT;
        testCase.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
        testCase.blockOrder = ACL_SPARSE_ORDER_COL;
        testCase.alpha = 0.0F;
        testCase.beta = -0.5F;
        testCase.specialDenseValues = true;
        testCase.randomValues = true;
        testCase.guardedRepeat = true;
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        EXPECT_TRUE(VerifyRealRun(result, testCase.outputType, testCase.name));
    }
    BsrComplexCase testCase;
    testCase.name = "GUARDED_ALPHA_ZERO_COMPLEX";
    testCase.blockSize = 64;
    testCase.k = 128;
    testCase.blockOrder = ACL_SPARSE_ORDER_COL;
    testCase.alpha = {0.0F, 0.0F};
    testCase.beta = {-0.5F, 0.25F};
    testCase.specialDenseValues = true;
    testCase.randomValues = true;
    testCase.guardedRepeat = true;
    SCOPED_TRACE(testCase.name);
    const auto result = RunBsrComplexCase(stream(), testCase);
    ASSERT_TRUE(StagesSucceeded(result.stages));
    EXPECT_TRUE(VerifyComplexRun(result, testCase.name));
}

TEST_F(SddmmExtendedNpuTest, BsrGuardedEmptyOutputUsesExactWorkspaceQuery)
{
    for (const int64_t blockRows : {1, 0}) {
        BsrRealCase testCase;
        testCase.name = "GUARDED_BSR_ZERO_NNZ_ROWS" + std::to_string(blockRows);
        testCase.blockRows = blockRows;
        testCase.blockSize = 16;
        testCase.k = 32;
        testCase.zeroBasedOffsets.assign(static_cast<size_t>(blockRows + 1), 0);
        testCase.zeroBasedColumns.clear();
        testCase.randomValues = true;
        testCase.guardedRepeat = true;
        SCOPED_TRACE(testCase.name);
        const auto result = RunBsrRealCase(stream(), testCase);
        ASSERT_TRUE(StagesSucceeded(result.stages));
        if (blockRows == 0) {
            EXPECT_EQ(result.stages.workspaceSize, 0U);
        }
        EXPECT_TRUE(result.actual.empty());
        EXPECT_TRUE(result.golden.empty());
    }
}

TEST_F(SddmmExtendedNpuTest, BsrRealBlock4K32MatchesGenericBroadcastBatch)
{
    const std::pair<aclDataType, aclDataType> types[] = {
        {ACL_FLOAT, ACL_FLOAT}, {ACL_FLOAT16, ACL_FLOAT16},
        {ACL_FLOAT16, ACL_FLOAT}, {ACL_BF16, ACL_BF16}, {ACL_BF16, ACL_FLOAT}};
    int ordinal = 0;
    for (const auto &type : types) {
        for (int layout = 0; layout < 2; ++layout) {
            for (int base = 0; base < 2; ++base) {
                for (int scalars = 0; scalars < 3; ++scalars) {
                    ASSERT_NO_FATAL_FAILURE(CheckBlock4GenericBroadcast(
                        stream(), type, layout, base, scalars, ordinal));
                }
            }
        }
    }
}

TEST_F(SddmmExtendedNpuTest, BsrRealBlock4K32TailsAndCoreBoundary)
{
    const std::pair<aclDataType, aclDataType> types[] = {
        {ACL_FLOAT, ACL_FLOAT}, {ACL_FLOAT16, ACL_FLOAT16},
        {ACL_FLOAT16, ACL_FLOAT}, {ACL_BF16, ACL_BF16}, {ACL_BF16, ACL_FLOAT}};
    // Odd block counts exercise partial thread groups. The larger cases
    // cover multiple cores while preserving independent output ownership.
    const int32_t blocks[] = {17, 19, 21, 56, 57};
    int ordinal = 0;
    for (int type = 0; type < 5; ++type) {
        for (int mode = 0; mode < 2; ++mode) {
            const int32_t count = blocks[(type + 2 * mode) % 5];
            ASSERT_NO_FATAL_FAILURE(CheckBsr4TailCase(
                stream(), types[type], count, ordinal++, mode != 0));
        }
    }
    for (int mode = 0; mode < 2; ++mode) {
        ASSERT_NO_FATAL_FAILURE(CheckBsr4TailCase(
            stream(), types[3], 56, ordinal++, mode != 0, 0));
        ASSERT_NO_FATAL_FAILURE(CheckBsr4TailCase(
            stream(), types[2], 17, ordinal++, mode != 0, 2));
    }
    // Tiny outputs and the exact thread-group boundary exercise one core.
    for (const int32_t count : {1, 3, 5, 16}) {
        ASSERT_NO_FATAL_FAILURE(CheckBsr4TailCase(
            stream(), types[3], count, ordinal++, false));
    }

    // Derive the core-cap boundary from the actual device and launch limit.
    // Keep the existing 1201-block test unchanged.
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ASSERT_NE(platform, nullptr);
    const uint32_t coreCount = platform->GetCoreNumAiv();
    ASSERT_GT(coreCount, 0U);
    ASSERT_LE(coreCount, static_cast<uint32_t>(
        (std::numeric_limits<int32_t>::max() - 16) / kSddmmThreadsPerBlock));
    const int32_t boundaryBlocks = static_cast<int32_t>(
        static_cast<uint64_t>(coreCount) * kSddmmThreadsPerBlock / 16U);
    for (const int32_t delta : {-1, 0, 1}) {
        SCOPED_TRACE("cores=" + std::to_string(coreCount) + " delta=" + std::to_string(delta));
        ASSERT_NO_FATAL_FAILURE(CheckBsr4TailCase(
            stream(), types[3], boundaryBlocks + delta, ordinal++, delta > 0));
    }
    EXPECT_EQ(ordinal, 21);
}

TEST_F(SddmmExtendedNpuTest, BsrWorkspaceDeviceToHostScalarsKeepsQueriedCapacity)
{
    BsrRealCase testCase;
    testCase.name = "BSR_WORKSPACE_DEVICE_TO_HOST";
    testCase.blockSize = 16;
    testCase.k = 64;
    testCase.deviceScalars = true;
    const int64_t m = testCase.blockRows * testCase.blockSize;
    const int64_t n = testCase.blockCols * testCase.blockSize;
    const int64_t outputs = static_cast<int64_t>(testCase.zeroBasedColumns.size()) *
        testCase.blockSize * testCase.blockSize;
    const auto x = MakeRealValues(1, m * testCase.k, 0.125F);
    const auto y = MakeRealValues(1, testCase.k * n, -0.0625F);
    const auto initial = MakeRealValues(1, outputs, 0.03125F);
    BsrWorkspaceFixture fixture(stream());
    fixture.Initialize(testCase, ACL_FLOAT, x, y, initial);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    fixture.guardedWorkspace = SddmmGuardedBuffer::copyFrom(nullptr, fixture.stages.workspaceSize);

    RealRunResult result;
    result.golden = BsrRealGolden(testCase, x, y, initial);
    ASSERT_TRUE(fixture.Run());
    result.actual = UnpackRealValues(fixture.dValues.snapshot(), initial.size(), ACL_FLOAT);
    ASSERT_TRUE(VerifyRealRun(result, ACL_FLOAT, testCase.name));

    fixture.SetScalars(stream(), false, testCase.alpha, testCase.beta);
    // Changing pointer mode must not enable Cube on the existing 64-byte buffer,
    // whether Execute reuses prepared state or Preprocess explicitly rebuilds it.
    for (const bool preprocess : {false, true}) {
        SCOPED_TRACE(preprocess);
        ASSERT_TRUE(fixture.Run(preprocess));
        result.actual = UnpackRealValues(fixture.dValues.snapshot(), initial.size(), ACL_FLOAT);
        ASSERT_TRUE(VerifyRealRun(result, ACL_FLOAT, testCase.name));
    }
    // A fresh query now needs Cube scratch: the earlier small-capacity runs
    // genuinely exercised the capacity restriction rather than an ineligible shape.
    ASSERT_TRUE(fixture.Query());
    ASSERT_GT(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
}

TEST_F(SddmmExtendedNpuTest, BsrWorkspaceAlignedComplexPointersKeepsQueriedCapacity)
{
    BsrComplexCase testCase;
    testCase.name = "BSR_WORKSPACE_COMPLEX_ALIGNMENT";
    testCase.k = 64;
    const int64_t m = testCase.blockRows * testCase.blockSize;
    const int64_t n = testCase.blockCols * testCase.blockSize;
    const auto x = MakeComplexValues(static_cast<size_t>(m * testCase.k), 0.125F);
    const auto y = MakeComplexValues(static_cast<size_t>(testCase.k * n), -0.0625F);
    const auto initial = MakeComplexValues(testCase.zeroBasedColumns.size() *
        static_cast<size_t>(testCase.blockSize * testCase.blockSize), 0.03125F);
    BsrWorkspaceFixture fixture(stream());
    fixture.Initialize(testCase, ACL_COMPLEX64, x, y, initial, sizeof(aclsparseComplex));
    constexpr auto alignment = sddmm_arch35::kSddmmComplexBsrCubeInputAlignment;
    ASSERT_NE(reinterpret_cast<uintptr_t>(fixture.dX.get()) % alignment, 0U);
    ASSERT_NE(reinterpret_cast<uintptr_t>(fixture.dY.get()) % alignment, 0U);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    fixture.guardedWorkspace = SddmmGuardedBuffer::copyFrom(nullptr, fixture.stages.workspaceSize);

    ComplexRunResult result;
    result.goldenParts = BsrComplexGoldenParts(testCase, x, y, initial);
    ASSERT_TRUE(fixture.Run());
    result.actualParts = UnpackRealValues(fixture.dValues.snapshot(), initial.size() * 2U, ACL_FLOAT);
    ASSERT_TRUE(VerifyComplexRun(result, testCase.name));

    auto alignedX = SddmmGuardedBuffer::copyFrom(x.data(), x.size() * sizeof(aclsparseComplex));
    auto alignedY = SddmmGuardedBuffer::copyFrom(y.data(), y.size() * sizeof(aclsparseComplex));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(alignedX.get()) % alignment, 0U);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(alignedY.get()) % alignment, 0U);
    ASSERT_EQ(aclsparseDnMatSetValues(fixture.matX.get(), alignedX.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseDnMatSetValues(fixture.matY.get(), alignedY.get()), ACL_SPARSE_STATUS_SUCCESS);
    for (const bool preprocess : {false, true}) {
        SCOPED_TRACE(preprocess);
        ASSERT_TRUE(fixture.Run(preprocess));
        CheckGuardedBuffers(fixture.dValues, fixture.guardedWorkspace, {&alignedX, &alignedY});
        result.actualParts = UnpackRealValues(fixture.dValues.snapshot(), initial.size() * 2U, ACL_FLOAT);
        ASSERT_TRUE(VerifyComplexRun(result, testCase.name));
    }
    ASSERT_TRUE(fixture.Query());
    ASSERT_GT(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    CheckGuardedBuffers(fixture.dValues, fixture.guardedWorkspace, {&alignedX, &alignedY});
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceQueryRejectsIneligiblePatterns)
{
    const CsrWorkspacePattern patterns[] = {CsrWorkspacePattern::REGULAR,
        CsrWorkspacePattern::PERTURBED, CsrWorkspacePattern::WEAK,
        CsrWorkspacePattern::UNALIGNED};
    for (const auto pattern : patterns) {
        SCOPED_TRACE(static_cast<int>(pattern));
        CsrWorkspaceFixture fixture(stream(), pattern);
        ASSERT_TRUE(fixture.Query());
        const size_t expected = pattern == CsrWorkspacePattern::REGULAR ?
            CsrWorkspaceFixture::cubeBytes : CsrWorkspaceFixture::genericBytes;
        ASSERT_EQ(fixture.stages.workspaceSize, expected);
        fixture.AllocateWorkspace();
        ASSERT_TRUE(fixture.Run());
        ASSERT_TRUE(fixture.Run());
    }
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceRawPatternUpgradeKeepsQueriedCapacity)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::PERTURBED);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    // The query's capacity applies even if indices change before the first
    // Preprocess, and remains valid through later in-place changes.
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::PERTURBED);
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Run());
    ASSERT_TRUE(fixture.Run());

    auto previous = std::move(fixture.workspace);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run(false));
    EXPECT_TRUE(previous.guardsIntact());
    fixture.workspace = std::move(previous);
    ASSERT_TRUE(fixture.Run(false));

    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::cubeBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceRegularToIrregularPreservesNumerics)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::cubeBytes);
    fixture.AllocateWorkspace();
    // A successful query must not cache pattern flags for the first Preprocess.
    fixture.ChangePattern(CsrWorkspacePattern::PERTURBED);
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::PERTURBED);
    ASSERT_TRUE(fixture.Run());
    ASSERT_TRUE(fixture.Run(false));

    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspacePointerSetterKeepsQueriedCapacity)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::WEAK);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR, true);
    ASSERT_TRUE(fixture.Run(false));
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::UNALIGNED, true);
    ASSERT_TRUE(fixture.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceNewQueryInvalidatesPreparedPattern)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Query());
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::ALIGNED_SHIFTED);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::cubeBytes);
    // Keep the address and capacity unchanged. Query must invalidate the
    // prepared first column even when the allocation key still matches.
    ASSERT_TRUE(fixture.Run(false));
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceReuseAcrossDescriptorsPreservesBounds)
{
    CsrWorkspaceFixture first(stream(), CsrWorkspacePattern::PERTURBED);
    CsrWorkspaceFixture second(stream(), CsrWorkspacePattern::UNALIGNED);
    ASSERT_TRUE(first.Query());
    ASSERT_TRUE(second.Query());
    ASSERT_EQ(first.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    ASSERT_EQ(second.stages.workspaceSize, first.stages.workspaceSize);
    first.AllocateWorkspace();
    second.AllocateWorkspace();
    ASSERT_TRUE(first.Run());
    ASSERT_TRUE(second.Run());
    std::swap(first.workspace, second.workspace);
    ASSERT_TRUE(first.Run(false));
    ASSERT_TRUE(second.Run(false));
    std::swap(first.workspace, second.workspace);
    ASSERT_TRUE(first.Run(false));
    ASSERT_TRUE(second.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceZeroToNonzeroAlphaPreservesBounds)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::PERTURBED);
    fixture.alpha = 0.0F;
    fixture.beta = 0.5F;
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    fixture.alpha = 0.875F;
    fixture.beta = 0.25F;
    ASSERT_TRUE(fixture.Run());
    fixture.alpha = 1.0F;
    fixture.beta = 0.0F;
    ASSERT_TRUE(fixture.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspacePointerModeChangeKeepsSmallCapacity)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR);
    fixture.SetDeviceScalars(true);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.SetDeviceScalars(false);
    ASSERT_TRUE(fixture.Run());
    fixture.alpha = -0.625F;
    fixture.beta = 0.375F;
    fixture.SetDeviceScalars(true);
    ASSERT_TRUE(fixture.Run(false));
    fixture.SetDeviceScalars(false);
    ASSERT_TRUE(fixture.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspacePointerModeChangeKeepsLargeCapacity)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::cubeBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.SetDeviceScalars(true);
    ASSERT_TRUE(fixture.Run());
    fixture.SetDeviceScalars(false);
    ASSERT_TRUE(fixture.Run(false));
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceLayoutChangeRequiresNewQuery)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::WEAK);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    fixture.ChangeYToColumnMajor();
    EXPECT_EQ(fixture.Preprocess(), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(fixture.Execute(), ACL_SPARSE_STATUS_INVALID_VALUE);
    fixture.CheckStorage();

    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, static_cast<size_t>(kSddmmWorkspaceBytes));
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceQueryChecksLastRowAndFreshPreprocess)
{
    // A long pattern must be checked through its final row. Base one and
    // cyclic wrap must retain Cube eligibility.
    constexpr int64_t rows = 144;
    const size_t cubeBytes = kSddmmWorkspaceBytes + rows * 16U * 320U / 16U * sizeof(float);
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR,
        rows, ACL_SPARSE_INDEX_BASE_ONE);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, cubeBytes);
    fixture.AllocateWorkspace();
    // Query cannot substitute its classification for the first Preprocess.
    fixture.PerturbRow(rows - 1);
    ASSERT_TRUE(fixture.Run());
    EXPECT_EQ(fixture.PatternFlags(), SDDMM_PATTERN_FLAG_NONE);

    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    ASSERT_TRUE(fixture.Run());
    EXPECT_NE(fixture.PatternFlags() & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    // The same smaller allocation remains safe after an in-place upgrade.
    fixture.PerturbRow(rows - 1);
    ASSERT_TRUE(fixture.Run());
    EXPECT_EQ(fixture.PatternFlags(), SDDMM_PATTERN_FLAG_NONE);

    // Each half still has stride 17 internally, but their global origins
    // differ. Query must compare every partition against row zero.
    fixture.ChangePattern(CsrWorkspacePattern::REGULAR);
    fixture.ShiftRows(rows / 2, 16);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    EXPECT_NE(fixture.PatternFlags() & SDDMM_PATTERN_FLAG_CSR_SORTED_CYCLIC_64, 0U);
    EXPECT_EQ(fixture.PatternFlags() & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
}

TEST_F(SddmmExtendedNpuTest, CsrWorkspaceQueryRejectsColumnsAcrossValidationBlocks)
{
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR,
        144, ACL_SPARSE_INDEX_BASE_ONE);
    ASSERT_TRUE(fixture.Query());
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Prepare());
    const size_t positions[] = {4095U, 4096U, fixture.ColumnCount() - 1U};
    for (const size_t position : positions) {
        SCOPED_TRACE(position);
        const int32_t original = fixture.Column(position);
        fixture.SetColumn(position, static_cast<int32_t>(CsrWorkspaceFixture::n) + 1);
        // A failing Preprocess must discard the previous valid prepared state.
        ASSERT_NO_FATAL_FAILURE(fixture.CheckRejectedPattern());
        ASSERT_TRUE(fixture.Query());
        EXPECT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
        fixture.AllocateWorkspace();
        fixture.SetColumn(position, original);
        ASSERT_TRUE(fixture.Prepare());
        EXPECT_NE(fixture.PatternFlags() & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    }
    ASSERT_TRUE(fixture.Run());
}

TEST_F(SddmmExtendedNpuTest, CsrValidationChecksOffsetBlockAdjacencyAndEndpoint)
{
    // Check an interior adjacent pair far from the first row as well as
    // the final offset of a long pattern.
    constexpr int64_t rows = 4112;
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR,
        rows, ACL_SPARSE_INDEX_BASE_ONE);
    ASSERT_TRUE(fixture.Query());
    ASSERT_GT(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Prepare());
    const size_t positions[] = {4096U, static_cast<size_t>(rows)};
    for (const size_t position : positions) {
        SCOPED_TRACE(position);
        const int32_t original = fixture.Offset(position);
        fixture.SetOffset(position, position == 4096U ?
            fixture.Offset(position - 1U) - 1 : original - 1);
        ASSERT_NO_FATAL_FAILURE(fixture.CheckRejectedPattern());
        ASSERT_TRUE(fixture.Query());
        EXPECT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
        fixture.AllocateWorkspace();
        fixture.SetOffset(position, original);
        ASSERT_TRUE(fixture.Prepare());
        EXPECT_NE(fixture.PatternFlags() & SDDMM_PATTERN_FLAG_CSR_CYCLIC_64_STRIDE_17, 0U);
    }
}

TEST_F(SddmmExtendedNpuTest, CsrValidationChecksUnalignedColumnTailBaseOne)
{
    // The final row has 63 entries: an invalid column only at the end must
    // still be rejected by generic Preprocess.
    CsrWorkspaceFixture fixture(stream(), CsrWorkspacePattern::REGULAR,
        129, ACL_SPARSE_INDEX_BASE_ONE, 63);
    ASSERT_EQ(fixture.ColumnCount(), 8255U);
    ASSERT_TRUE(fixture.Query());
    ASSERT_EQ(fixture.stages.workspaceSize, CsrWorkspaceFixture::genericBytes);
    fixture.AllocateWorkspace();
    ASSERT_TRUE(fixture.Run());
    EXPECT_EQ(fixture.PatternFlags(), SDDMM_PATTERN_FLAG_NONE);
    const size_t last = fixture.ColumnCount() - 1U;
    const int32_t original = fixture.Column(last);
    fixture.SetColumn(last, 0); // Below the base-one lower bound, only in the tail.
    ASSERT_NO_FATAL_FAILURE(fixture.CheckRejectedPattern());
    fixture.SetColumn(last, original);
    ASSERT_TRUE(fixture.Run());
    EXPECT_EQ(fixture.PatternFlags(), SDDMM_PATTERN_FLAG_NONE);
}
} // namespace sddmm_extended
} // namespace sparse_test

using namespace sparse_test;
using namespace sparse_test::sddmm_extended;

class SddmmTest : public testing::TestWithParam<SddmmTestParam> {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    SddmmTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

class SddmmExceptionTest : public testing::Test {
public:
    static void SetUpTestSuite() {
        env_ = std::make_unique<AclEnvScope>();
    }

    static void TearDownTestSuite() {
        env_.reset();
    }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;

    aclrtStream stream_ = nullptr;
    std::unique_ptr<HandleManager> handle_;
    SddmmCsr csrC_;
    int nnz_ = 0;

    // RAII device buffers and descriptors (auto-released on fixture destruction)
    DeviceBuffer dRowOff_;
    DeviceBuffer dColInd_;
    DeviceBuffer dVals_;
    DeviceBuffer dX_;
    DeviceBuffer dY_;
    DeviceBuffer dBuffer_;
    DnMatManager matX_;
    DnMatManager matY_;
    SpMatManager matC_;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    size_t bufSize_ = 0;

    void SetUp() override {
        stream_ = env_->stream();
        handle_ = std::make_unique<HandleManager>();
        handle_->setStream(stream_);

        // Small 4×4 CSR for valid-parameter baseline
        csrC_ = MakeSddmmSparsity(4, 4, 0.5, -1.0, 1.0, 42);
        nnz_ = static_cast<int>(csrC_.nnz);

        std::vector<float> xFp32(16, 1.0f);
        std::vector<float> yFp32(16, 1.0f);
        std::vector<float> cInitFp32(nnz_);
        for (int i = 0; i < nnz_; i++) {
            cInitFp32[static_cast<size_t>(i)] = static_cast<float>(csrC_.values[static_cast<size_t>(i)]);
        }

        dRowOff_ = DeviceBuffer::copyFrom(csrC_.rowOffsets.data(), 5 * sizeof(int32_t));
        if (nnz_ > 0) {
            dColInd_ = DeviceBuffer::copyFrom(csrC_.colIndices.data(),
                                              static_cast<size_t>(nnz_) * sizeof(int32_t));
            dVals_ = DeviceBuffer::copyFrom(cInitFp32.data(),
                                            static_cast<size_t>(nnz_) * sizeof(float));
        }
        dX_ = DeviceBuffer::copyFrom(xFp32.data(), 16 * sizeof(float));
        dY_ = DeviceBuffer::copyFrom(yFp32.data(), 16 * sizeof(float));

        // X/Y: 4×4 row-major const dense; C: 4×4 CSR (FP32)
        matX_ = DnMatManager::createConst(4, 4, 4, dX_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matY_ = DnMatManager::createConst(4, 4, 4, dY_.raw(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
        matC_ = SpMatManager::createCsr(4, 4, nnz_, dRowOff_.get(), dColInd_.get(),
                                        dVals_.get(), ACL_SPARSE_INDEX_32I,
                                        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                        ACL_FLOAT);

        // Query buffer size for the valid baseline (may fail if operator not
        // yet implemented; defensive fallback keeps E7 testable).
        bufSize_ = 0;
        auto ret = aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bufSize_);
        if (ret != ACL_SPARSE_STATUS_SUCCESS || bufSize_ == 0) {
            bufSize_ = 16;
        }
        dBuffer_ = DeviceBuffer::alloc(bufSize_);
    }
};

using SddmmZeroDimensionTest = sddmm_extended::SddmmExtendedNpuTest;

TEST_P(SddmmTest, SddmmFunctional) {
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ====" << p.description
              << " m=" << p.m << " k=" << p.k << " n=" << p.n
              << " ratio=" << p.sparsity_ratio
              << " alpha=" << p.alpha << " beta=" << p.beta
              << " dtype=" << p.dtype
              << " order_x=" << p.order_x << " order_y=" << p.order_y
              << " value_range=[" << p.value_lo << "," << p.value_hi << "]"
              << " seed=" << p.random_seed << "\n";

    // --- Step 1+2: deterministic sparsity pattern + X/Y (FP64) ---------------
    SddmmInputs inputs = GenerateSddmmInputs(p);
    int64_t nnz = inputs.nnz;
    std::cout << "  nnz=" << nnz << "\n";

    // --- Step 3: parse dtype / op / order enums + golden reference -----------
    aclDataType dtype = ParseDtype(p.dtype);
    aclsparseOperation_t opX = ParseOperation(p.op_x);
    aclsparseOperation_t opY = ParseOperation(p.op_y);
    aclsparseOrder_t orderX = ParseOrder(p.order_x);
    aclsparseOrder_t orderY = ParseOrder(p.order_y);
    std::vector<double> expectFp64 = ComputeGoldenExpect(p, inputs.csrC, inputs.Xf64,
                                                          inputs.Yf64, dtype, opX, opY);

    // --- Step 4: parse remaining ACL enums -----------------------------------
    aclDataType computeType = ParseDtype(p.compute_type);
    aclsparseSDDMMAlg_t alg = ParseSddmmAlg(p.alg);

    // --- Step 5: NPU three-stage workflow ------------------------------------
    SddmmNpuResult npuResult = RunNpuSddmm(p, stream_, inputs.csrC, inputs.Xf64,
                                            inputs.Yf64, dtype, computeType, alg,
                                            opX, opY, orderX, orderY, nnz);

    // --- Step 6: check return codes ------------------------------------------
    ASSERT_EQ(npuResult.bufferSizeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "BufferSize stage failed";
    ASSERT_EQ(npuResult.preprocessRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Preprocess stage failed";
    ASSERT_EQ(npuResult.executeRet, ACL_SPARSE_STATUS_SUCCESS)
        << "Execute stage failed";

    // --- Step 7: precision verification (MIXED_TOLERANCE, dtype-driven) ------
    if (nnz > 0) {
        ASSERT_EQ(npuResult.valuesOut.size(), static_cast<size_t>(nnz));
    }
    VerifySddmmPrecision(p.case_name, npuResult, expectFp64, dtype, nnz);

    std::cout << "[" << p.case_name << "] PASSED (nnz=" << nnz << ")\n";
}

INSTANTIATE_TEST_CASE_P(
    SddmmCases,
    SddmmTest,
    testing::ValuesIn(GetCasesFromCsv<SddmmTestParam>("sddmm_test.csv")),
    [](const testing::TestParamInfo<SddmmTestParam>& info) {
        return info.param.case_name;
    }
);

TEST_F(SddmmZeroDimensionTest, MZero)
{
    ASSERT_NO_FATAL_FAILURE(CheckZeroOutputDimension(stream(), true));
}

TEST_F(SddmmZeroDimensionTest, NZero)
{
    ASSERT_NO_FATAL_FAILURE(CheckZeroOutputDimension(stream(), false));
}

TEST_F(SddmmZeroDimensionTest, KZero)
{
    for (const float beta : {-0.5F, 0.0F}) {
        SCOPED_TRACE(beta);
        ASSERT_NO_FATAL_FAILURE(CheckZeroReduction(stream(), beta));
    }
}

TEST_F(SddmmExceptionTest, NullHandle) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        nullptr, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SddmmExceptionTest, NullMatX) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, nullptr, matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, NullMatY) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), nullptr, &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, NullMatC) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, nullptr,
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, NullAlpha) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        nullptr, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, NullBeta) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), nullptr, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, NullExternalBuffer) {
    auto ret = aclsparseSDDMM(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, nullptr);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, DimMismatch) {
    // Y with k=2 (mismatches X k=4): n=4, k=2 row-major
    std::vector<float> ySmall(8, 1.0f);
    auto dYSmall = DeviceBuffer::copyFrom(ySmall.data(), 8 * sizeof(float));
    auto matYSmall = DnMatManager::createConst(4, 2, 2, dYSmall.raw(),
                                               ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matYSmall.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_TRUE(ret == ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED ||
                ret == ACL_SPARSE_STATUS_INVALID_VALUE)
        << "expected MATRIX_TYPE_NOT_SUPPORTED or INVALID_VALUE, got " << ret;
}

TEST_F(SddmmExceptionTest, UnsupportedComputeType) {
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_DOUBLE, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(SddmmExceptionTest, NullCsrPtrs) {
    auto matCNullPtrs = SpMatManager::createCsr(4, 4, 0, nullptr, dColInd_.get(),
                                                 dVals_.get(), ACL_SPARSE_INDEX_32I,
                                                 ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                                 ACL_FLOAT);
    size_t bs = 0;
    auto ret = aclsparseSDDMMBufferSize(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matCNullPtrs.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &bs);
    EXPECT_EQ(ret, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST_F(SddmmExceptionTest, DenseLeadingDimension64BitBounds) {
    // arch35 carries ld in int64_t. Querying a representable span must not
    // truncate it to int32_t; overflowing element/byte spans remain invalid.
    // BufferSize only inspects metadata, so no large Device allocation is needed.
    const int64_t leadingDimensions[] = {
        static_cast<int64_t>(INT32_MAX) + 1,
        std::numeric_limits<int64_t>::max() / 4,
        std::numeric_limits<int64_t>::max()};
    const aclsparseStatus_t expected[] = {
        ACL_SPARSE_STATUS_SUCCESS, ACL_SPARSE_STATUS_INVALID_VALUE,
        ACL_SPARSE_STATUS_INVALID_VALUE};
    for (const auto order : {ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL}) {
        for (size_t i = 0; i < 3; ++i) {
            auto wide = DnMatManager::createConst(
                4, 4, leadingDimensions[i], dX_.raw(), ACL_FLOAT, order);
            for (const bool useWideX : {false, true}) {
                size_t size = 0;
                EXPECT_EQ(aclsparseSDDMMBufferSize(
                    handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE,
                    ACL_SPARSE_OP_NON_TRANSPOSE, &alpha_,
                    useWideX ? wide.cget() : matX_.cget(),
                    useWideX ? matY_.cget() : wide.cget(), &beta_, matC_.get(),
                    ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &size), expected[i]);
            }
        }
    }
}

TEST_F(SddmmExceptionTest, DenseBatchStrideRequiresFullPaddedSpan) {
    ASSERT_NO_FATAL_FAILURE(CheckDenseBatchStrideRequiresFullPaddedSpan(
        handle_.get(), alpha_, beta_, dBuffer_));
}

TEST_F(SddmmExceptionTest, CsrElementStridesRejectedAndUnitLayoutRestored) {
    const int64_t strides[][3] = {{2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
    for (const auto &stride : strides) {
        ASSERT_EQ(aclsparseCsrSetStrides(matC_.get(), stride[0], stride[1], stride[2]),
                  ACL_SPARSE_STATUS_SUCCESS);
        size_t size = 0;
        EXPECT_EQ(aclsparseSDDMMBufferSize(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, &size), ACL_SPARSE_STATUS_NOT_SUPPORTED);
        EXPECT_EQ(aclsparseSDDMMPreprocess(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer_.get()), ACL_SPARSE_STATUS_NOT_SUPPORTED);
        EXPECT_EQ(aclsparseSDDMM(
            handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
            &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
            ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer_.get()), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    }
    ASSERT_EQ(aclsparseCsrSetStrides(matC_.get(), 1, 1, 1), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclsparseSDDMM(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer_.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    std::vector<float> actual(static_cast<size_t>(nnz_));
    dVals_.copyToHost(actual.data(), actual.size() * sizeof(float));
    for (const float value : actual) {
        EXPECT_FLOAT_EQ(value, 4.0F);
    }
}

TEST_F(SddmmExceptionTest, CsrStrideSetterInvalidatesPreparedState) {
    ASSERT_EQ(aclsparseSDDMMPreprocess(
        handle_->get(), ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE,
        &alpha_, matX_.cget(), matY_.cget(), &beta_, matC_.get(),
        ACL_FLOAT, ACL_SPARSE_SDDMM_ALG_DEFAULT, dBuffer_.get()), ACL_SPARSE_STATUS_SUCCESS);
    auto *descr = matC_.get();
    const auto generation = descr->structureGeneration;
    ASSERT_TRUE(descr->sddmmPrepared.prepared);
    descr->activeBuffer = dBuffer_.get();
    descr->activeSpmvBuffer = dBuffer_.get();
    EXPECT_EQ(aclsparseCsrSetStrides(descr, 0, 1, 1), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr->activeBuffer, dBuffer_.get());
    EXPECT_EQ(descr->activeSpmvBuffer, dBuffer_.get());
    EXPECT_TRUE(descr->sddmmPrepared.prepared);
    EXPECT_EQ(descr->structureGeneration, generation);
    ASSERT_EQ(aclsparseCsrSetStrides(descr, 1, 1, 1), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(descr->activeBuffer, nullptr);
    EXPECT_EQ(descr->activeSpmvBuffer, nullptr);
    EXPECT_TRUE(descr->sddmmPrepared.prepared);
    EXPECT_EQ(descr->structureGeneration, generation);
    descr->activeBuffer = dBuffer_.get();
    descr->activeSpmvBuffer = dBuffer_.get();
    ASSERT_EQ(aclsparseCsrSetStrides(descr, 1, 2, 1), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(descr->activeBuffer, nullptr);
    EXPECT_EQ(descr->activeSpmvBuffer, nullptr);
    EXPECT_FALSE(descr->sddmmPrepared.prepared);
    EXPECT_GT(descr->structureGeneration, generation);
    int64_t offsets = 0, columns = 0, values = 0;
    ASSERT_EQ(aclsparseCsrGetStrides(descr, &offsets, &columns, &values), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(offsets, 1);
    EXPECT_EQ(columns, 2);
    EXPECT_EQ(values, 1);
}

TEST(SddmmDescriptorApiTest, CreateBsrAndConstBsrAcceptSupportedGeometry) {
    const int64_t blockSizes[] = {2, 4, 8, 16, 32, 64, 128};
    const aclsparseIndexBase_t indexBases[] = {
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE};
    const aclsparseOrder_t orders[] = {
        ACL_SPARSE_ORDER_ROW, ACL_SPARSE_ORDER_COL};

    for (const int64_t blockSize : blockSizes) {
        for (const auto indexBase : indexBases) {
            for (const auto order : orders) {
                SCOPED_TRACE(testing::Message()
                             << "blockSize=" << blockSize
                             << ", indexBase=" << indexBase
                             << ", order=" << order);
                const int32_t base =
                    indexBase == ACL_SPARSE_INDEX_BASE_ZERO ? 0 : 1;
                std::vector<int32_t> rowOffsets = {
                    base, base + 2, base + 3};
                std::vector<int32_t> colIndices = {
                    base, base + 1, base + 2};
                std::vector<float> values(
                    static_cast<size_t>(3 * blockSize * blockSize), 1.0f);

                BsrCreateParams p;
                p.rowBlockSize = blockSize;
                p.colBlockSize = blockSize;
                p.rowOffsets = rowOffsets.data();
                p.colIndices = colIndices.data();
                p.values = values.data();
                p.indexBase = indexBase;
                p.order = order;
                ExpectMutableAndConstBsrCreate(
                    p, ACL_SPARSE_STATUS_SUCCESS);
            }
        }
    }

    // Empty BSR matrices still own a row-offset endpoint, while their empty
    // columns and values arrays may be represented by null pointers.
    std::vector<int32_t> emptyRowOffsets = {0, 0, 0};
    BsrCreateParams empty;
    empty.blockNnz = 0;
    empty.rowOffsets = emptyRowOffsets.data();
    ExpectMutableAndConstBsrCreate(empty, ACL_SPARSE_STATUS_SUCCESS);
}

TEST(SddmmDescriptorApiTest, CreateBsrAndConstBsrRejectInvalidBlockGeometry) {
    std::vector<int32_t> rowOffsets = {0, 2, 3};
    std::vector<int32_t> colIndices = {0, 1, 2};
    std::vector<float> values(12, 1.0f);
    BsrCreateParams valid;
    valid.rowOffsets = rowOffsets.data();
    valid.colIndices = colIndices.data();
    valid.values = values.data();

    const auto expectGeometry = [&](int64_t rowBlockSize,
                                    int64_t colBlockSize,
                                    aclsparseStatus_t expected) {
        SCOPED_TRACE(testing::Message()
                     << "rowBlockSize=" << rowBlockSize
                     << ", colBlockSize=" << colBlockSize);
        BsrCreateParams p = valid;
        p.rowBlockSize = rowBlockSize;
        p.colBlockSize = colBlockSize;
        ExpectMutableAndConstBsrCreate(p, expected);
    };

    expectGeometry(0, 2, ACL_SPARSE_STATUS_INVALID_VALUE);
    expectGeometry(2, 0, ACL_SPARSE_STATUS_INVALID_VALUE);
    expectGeometry(-2, 2, ACL_SPARSE_STATUS_INVALID_VALUE);
    expectGeometry(2, -2, ACL_SPARSE_STATUS_INVALID_VALUE);
    expectGeometry(2, 4, ACL_SPARSE_STATUS_INVALID_VALUE);
    expectGeometry(3, 3, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST(SddmmDescriptorApiTest, CreateBsrAndConstBsrRejectInvalidMetadataAndPointers) {
    std::vector<int32_t> rowOffsets = {0, 2, 3};
    std::vector<int32_t> colIndices = {0, 1, 2};
    std::vector<float> values(12, 1.0f);
    BsrCreateParams valid;
    valid.rowOffsets = rowOffsets.data();
    valid.colIndices = colIndices.data();
    valid.values = values.data();

    EXPECT_EQ(CreateMutableBsr(valid, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(CreateConstBsr(valid, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    BsrCreateParams p = valid;
    p.rowOffsetsType = ACL_SPARSE_INDEX_64I;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_NOT_SUPPORTED);
    p = valid;
    p.colIndicesType = ACL_SPARSE_INDEX_64I;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_NOT_SUPPORTED);

    p = valid;
    p.indexBase = static_cast<aclsparseIndexBase_t>(-1);
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.order = static_cast<aclsparseOrder_t>(-1);
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);

    p = valid;
    p.rowOffsets = nullptr;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.colIndices = nullptr;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.values = nullptr;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST(SddmmDescriptorApiTest, CreateBsrAndConstBsrRejectNegativeOverflowAndI32Limit) {
    std::vector<int32_t> rowOffsets = {0, 2, 3};
    std::vector<int32_t> colIndices = {0, 1, 2};
    std::vector<float> values(12, 1.0f);
    BsrCreateParams valid;
    valid.rowOffsets = rowOffsets.data();
    valid.colIndices = colIndices.data();
    valid.values = values.data();

    BsrCreateParams p = valid;
    p.blockRows = -1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.blockCols = -1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.blockNnz = -1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);

    // The scalar shape multiplication must be checked before any allocation
    // or pointer arithmetic is attempted.
    p = valid;
    p.blockRows = std::numeric_limits<int64_t>::max();
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.blockCols = std::numeric_limits<int64_t>::max();
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);

    p = valid;
    p.blockRows = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
                  p.rowBlockSize + 1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_SUCCESS);
    p = valid;
    p.blockRows = std::numeric_limits<int32_t>::max();
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.blockCols = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);

    p = valid;
    p.blockNnz = static_cast<int64_t>(
                         std::numeric_limits<int32_t>::max()) +
                     1;
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
    p = valid;
    p.indexBase = ACL_SPARSE_INDEX_BASE_ONE;
    p.blockNnz = std::numeric_limits<int32_t>::max();
    ExpectMutableAndConstBsrCreate(p, ACL_SPARSE_STATUS_INVALID_VALUE);
}

TEST(SddmmDescriptorApiTest, DnMatStridedBatchRoundTripsBoundaryCounts) {
    constexpr int64_t physicalSpan = 8; // ROW: rows * ld = 2 * 4.
    std::vector<float> values(
        static_cast<size_t>(65535 * physicalSpan), 0.0f);
    auto mat = DnMatManager::create(
        2, 3, 4, values.data(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);

    ExpectDnMatBatchState(mat.cget(), 1, 0);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(mat.get(), 1, 0),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectDnMatBatchState(mat.cget(), 1, 0);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(mat.get(), 2, physicalSpan),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectDnMatBatchState(mat.cget(), 2, physicalSpan);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(mat.get(), 65535, physicalSpan),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectDnMatBatchState(mat.cget(), 65535, physicalSpan);
}

TEST(SddmmDescriptorApiTest, DnMatStridedBatchRejectsInvalidArguments) {
    std::vector<float> rowValues(16, 0.0f);
    auto rowMat = DnMatManager::create(
        2, 3, 4, rowValues.data(), ACL_FLOAT, ACL_SPARSE_ORDER_ROW);
    std::vector<float> colValues(24, 0.0f);
    auto colMat = DnMatManager::create(
        2, 3, 4, colValues.data(), ACL_FLOAT, ACL_SPARSE_ORDER_COL);

    EXPECT_EQ(aclsparseDnMatSetStridedBatch(colMat.get(), 2, 12),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectDnMatBatchState(colMat.cget(), 2, 12);

    EXPECT_EQ(aclsparseDnMatSetStridedBatch(nullptr, 1, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(rowMat.get(), 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(rowMat.get(), 65536, 8),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(rowMat.get(), 2, -1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(rowMat.get(), 2, 7),
              ACL_SPARSE_STATUS_SUCCESS);
    ExpectDnMatBatchState(rowMat.cget(), 2, 7);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(colMat.get(), 2, 11),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(colMat.get(), 2, 10),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(rowMat.get(), 2, 6),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(colMat.get(), 2, 9),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectDnMatBatchState(colMat.cget(), 2, 10);
    EXPECT_EQ(aclsparseDnMatSetStridedBatch(
                  rowMat.get(), 65535,
                  std::numeric_limits<int64_t>::max()),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    int batchCount = 0;
    int64_t batchStride = 0;
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(
                  nullptr, &batchCount, &batchStride),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(
                  rowMat.cget(), nullptr, &batchStride),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseDnMatGetStridedBatch(
                  rowMat.cget(), &batchCount, nullptr),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    ExpectDnMatBatchState(rowMat.cget(), 2, 7);
}

TEST(SddmmDescriptorApiTest, BsrStridedBatchAcceptsSharedAndNonSharedPatterns) {
    constexpr int64_t valuesSpan = 12; // 3 blocks * 2 * 2 values.
    std::vector<int32_t> rowOffsets = {0, 2, 3, 0, 2, 3};
    std::vector<int32_t> colIndices = {0, 1, 2, 0, 1, 2};
    std::vector<float> values(
        static_cast<size_t>(65535 * valuesSpan), 0.0f);
    BsrCreateParams p;
    p.rowOffsets = rowOffsets.data();
    p.colIndices = colIndices.data();
    p.values = values.data();
    MutableSpMatGuard descr;
    ASSERT_EQ(CreateMutableBsr(p, descr.put()), ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 1, 0, 0, 0),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(
                  descr.get(), 2, 0, 0, valuesSpan),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(
                  descr.get(), 2, 3, 3, valuesSpan),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(
                  descr.get(), 65535, 0, 0, valuesSpan),
              ACL_SPARSE_STATUS_SUCCESS);
}

TEST(SddmmDescriptorApiTest, BsrStridedBatchRejectsOverlapBoundsAndWrongFormat) {
    std::vector<int32_t> rowOffsets = {0, 2, 3};
    std::vector<int32_t> colIndices = {0, 1, 2};
    std::vector<float> values(24, 0.0f);
    BsrCreateParams p;
    p.rowOffsets = rowOffsets.data();
    p.colIndices = colIndices.data();
    p.values = values.data();
    MutableSpMatGuard descr;
    ASSERT_EQ(CreateMutableBsr(p, descr.put()), ACL_SPARSE_STATUS_SUCCESS);

    EXPECT_EQ(aclsparseBsrSetStridedBatch(nullptr, 1, 0, 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 0, 0, 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 65536, 0, 0, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, -1, 0, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 0, -1, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 0, 0, -1),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 2, 0, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 0, 2, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 0, 0, 0),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(descr.get(), 2, 0, 0, 11),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(
                  descr.get(), 65535, 0, 0,
                  std::numeric_limits<int64_t>::max()),
              ACL_SPARSE_STATUS_INVALID_VALUE);

    auto csr = SpMatManager::createCsr(
        2, 3, 3, rowOffsets.data(), colIndices.data(), values.data(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    EXPECT_EQ(aclsparseBsrSetStridedBatch(csr.get(), 2, 0, 0, 12),
              ACL_SPARSE_STATUS_INVALID_VALUE);
}
