/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file gather_test.cpp
 * @brief GTest + CSV-driven test cases for aclsparseGather.
 *
 * Gather: X.values[i] = Y[X.indices[i] - idxBase] for i = 0 .. nnz-1.
 *
 * Test parameters are loaded from gather_test.csv (copied to build dir by CMake).
 * Verification uses the test framework's Verifier with appropriate precision mode.
 *
 * Entry point is shared via test/frame/test_main.cpp.
 */

#include <gtest/gtest.h>
#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "fill.h"
#include "test_common.h"
#include "gather_golden.h"
#include "gather_npu_wrapper.h"
#include "gather_param.h"
#include "gather_stride_test.h"
#include "aclsparse_descr_internal.h"

#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace sparse_test;

class GatherTest : public testing::TestWithParam<GatherTestParam> {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }

    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
    GatherTestParam param_;
    aclrtStream stream_ = nullptr;

    void SetUp() override
    {
        param_ = GetParam();
        stream_ = env_->stream();
    }
};

TEST_P(GatherTest, Gather)
{
    const auto& p = param_;

    std::cout << "==== " << p.case_name << " ==== vec_size=" << p.vec_size << " nnz=" << p.nnz << "\n";

    std::mt19937 rng(p.seed);
    HandleManager handle;
    handle.setStream(stream_);

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool pass = false;
    int match_count = 0;

    // c++17 don't support template lambda
    auto match_case = [&match_count, &p, &rng, &handle, &cfg, &pass](
                          aclDataType aclFloat_t, auto float_v, aclsparseIndexType_t aclIndex_t, auto int_v) -> void {
        using float_t = decltype(float_v);
        using int_t = decltype(int_v);
        if (p.value_type == aclFloat_t && p.idx_type == aclIndex_t) {
            match_count += 1;
            std::uniform_int_distribution<int_t> idxDist(p.idx_base, p.vec_size - 1 + p.idx_base);
            auto YHost = makeDense<float_t>(p.vec_size, -2.0, 2.0, p.seed);
            std::vector<int_t> indices(p.nnz);
            for (auto& i : indices) {
                i = idxDist(rng);
            }
            std::vector<float_t> golden = GatherGolden<float_t, int_t>(YHost, indices, p.idx_base);
            std::vector<float_t> output =
                GatherNpu<float_t, int_t>(handle, YHost, indices, p.value_type, p.idx_type, p.idx_base);
            // Verifier::verifyVector 仅支持 float32 类型
            pass = Verifier::verifyVector(
                std::vector<float>(output.begin(), output.end()), std::vector<float>(golden.begin(), golden.end()), cfg,
                p.case_name);
        }
    };
    // bf16 type don't exist on host
    match_case(ACL_FLOAT, float{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_FLOAT, float{}, ACL_SPARSE_INDEX_64I, int64_t{});
    // Gather is a bit-wise copy; uint16_t is the portable Host payload for fp16.
    match_case(ACL_FLOAT16, uint16_t{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_FLOAT16, uint16_t{}, ACL_SPARSE_INDEX_64I, int64_t{});
    match_case(ACL_BF16, uint16_t{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_BF16, uint16_t{}, ACL_SPARSE_INDEX_64I, int64_t{});
    match_case(ACL_COMPLEX64, uint64_t{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_COMPLEX64, uint64_t{}, ACL_SPARSE_INDEX_64I, int64_t{});
    match_case(ACL_DOUBLE, double{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_DOUBLE, double{}, ACL_SPARSE_INDEX_64I, int64_t{});
    ASSERT_EQ(match_count, 1);

    EXPECT_TRUE(pass) << "Gather verification failed";

    if (pass) {
        std::cout << "[" << p.case_name << "] PASSED\n";
    } else {
        std::cout << "[" << p.case_name << "] FAILED\n";
    }
}

INSTANTIATE_TEST_SUITE_P(
    GatherCases, GatherTest, testing::ValuesIn(GetCasesFromCsv<GatherTestParam>("gather_test.csv")),
    [](const testing::TestParamInfo<GatherTestParam>& info) { return info.param.case_name; });

class GatherContractTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

template <typename PayloadT>
void CheckBitExactGather(const std::vector<PayloadT>& yHost, aclDataType valueType,
                         aclsparseIndexBase_t base, aclrtStream stream)
{
    std::vector<int32_t> indices{5, 0, 3, 3, 1, 7, 2, 0};
    for (auto& index : indices) {
        index += static_cast<int32_t>(base);
    }
    const auto indicesBefore = indices;
    const auto yBefore = yHost;
    std::vector<PayloadT> expected(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        expected[i] = yHost[static_cast<size_t>(indices[i] - static_cast<int32_t>(base))];
    }

    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(PayloadT));
    auto dIndices = DeviceBuffer::copyFrom(indices.data(), indices.size() * sizeof(int32_t));
    auto dX = DeviceBuffer::alloc(indices.size() * sizeof(PayloadT));
    auto y = DnVecManager::createConst(static_cast<int64_t>(yHost.size()), dY.raw(), valueType);
    auto x = SpVecManager::create(static_cast<int64_t>(yHost.size()), static_cast<int64_t>(indices.size()),
                                  dIndices.get(), dX.get(), ACL_SPARSE_INDEX_32I, base, valueType);
    HandleManager handle;
    handle.setStream(stream);

    std::vector<PayloadT> first(indices.size());
    std::vector<PayloadT> second(indices.size());
    ASSERT_EQ(aclsparseGather(handle.get(), y.cget(), x.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    dX.copyToHost(first.data(), first.size() * sizeof(PayloadT));
    ASSERT_EQ(aclsparseGather(handle.get(), y.cget(), x.get()), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    dX.copyToHost(second.data(), second.size() * sizeof(PayloadT));

    std::vector<PayloadT> yAfter(yHost.size());
    std::vector<int32_t> indicesAfter(indices.size());
    dY.copyToHost(yAfter.data(), yAfter.size() * sizeof(PayloadT));
    dIndices.copyToHost(indicesAfter.data(), indicesAfter.size() * sizeof(int32_t));
    EXPECT_EQ(std::memcmp(first.data(), expected.data(), first.size() * sizeof(PayloadT)), 0);
    EXPECT_EQ(std::memcmp(second.data(), first.data(), first.size() * sizeof(PayloadT)), 0);
    EXPECT_EQ(std::memcmp(yAfter.data(), yBefore.data(), yAfter.size() * sizeof(PayloadT)), 0);
    EXPECT_EQ(indicesAfter, indicesBefore);
}

TEST_F(GatherContractTest, BitExactAllRequiredDtypesAndBases)
{
    const std::vector<uint16_t> fp16Bits{
        0x0000U, 0x8000U, 0x0001U, 0x3c00U, 0xbc00U, 0x7c00U, 0xfc00U, 0x7e01U};
    const std::vector<uint16_t> bf16Bits{
        0x0000U, 0x8000U, 0x0001U, 0x3f80U, 0xbf80U, 0x7f80U, 0xff80U, 0x7fc1U};
    const std::vector<uint32_t> fp32Bits{
        0x00000000U, 0x80000000U, 0x00000001U, 0x3f800000U,
        0xbf800000U, 0x7f800000U, 0xff800000U, 0x7fc01234U};
    // Each uint64_t stores {real float32 bits, imaginary float32 bits}.
    const std::vector<uint64_t> complex64Bits{
        0x0000000000000000ULL, 0x8000000080000000ULL, 0x3f800000bf800000ULL, 0xbf8000003f800000ULL,
        0x7f800000ff800000ULL, 0xff8000007f800000ULL, 0x7fc012343f000000ULL, 0x3f0000007fc05678ULL};

    for (auto base : {ACL_SPARSE_INDEX_BASE_ZERO, ACL_SPARSE_INDEX_BASE_ONE}) {
        CheckBitExactGather(fp16Bits, ACL_FLOAT16, base, env_->stream());
        CheckBitExactGather(bf16Bits, ACL_BF16, base, env_->stream());
        CheckBitExactGather(fp32Bits, ACL_FLOAT, base, env_->stream());
        CheckBitExactGather(complex64Bits, ACL_COMPLEX64, base, env_->stream());
    }
}

TEST_F(GatherContractTest, EmptyVectorsAreNoOp)
{
    HandleManager handle;
    handle.setStream(env_->stream());
    auto y = DnVecManager::createConst(0, nullptr, ACL_FLOAT);
    auto x = SpVecManager::create(0, 0, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                  ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
    EXPECT_EQ(aclsparseGather(handle.get(), y.cget(), x.get()), ACL_SPARSE_STATUS_SUCCESS);
}

struct GatherContractInputs {
    std::vector<uint32_t> yHost{1U, 2U, 3U, 4U};
    std::vector<int32_t> indices{0, 3};
    DeviceBuffer dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(uint32_t));
    DeviceBuffer dIndices = DeviceBuffer::copyFrom(indices.data(), indices.size() * sizeof(int32_t));
    DeviceBuffer dX = DeviceBuffer::alloc(indices.size() * sizeof(uint32_t));
    HandleManager handle;
    DnVecManager y = DnVecManager::createConst(4, dY.raw(), ACL_FLOAT);
    SpVecManager x = SpVecManager::create(4, 2, dIndices.get(), dX.get(), ACL_SPARSE_INDEX_32I,
                                          ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);
};

TEST_F(GatherContractTest, RejectsInvalidDescriptorObjects)
{
    GatherContractInputs input;
    input.handle.setStream(env_->stream());
    EXPECT_EQ(aclsparseGather(nullptr, input.y.cget(), input.x.get()), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseGather(input.handle.get(), nullptr, input.x.get()), ACL_SPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), nullptr), ACL_SPARSE_STATUS_INVALID_VALUE);

    const auto signature = input.y.get()->signature;
    input.y.get()->signature = 0;
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    input.y.get()->signature = signature;
}

TEST_F(GatherContractTest, RejectsInvalidTypesAndShape)
{
    GatherContractInputs input;
    input.handle.setStream(env_->stream());
    const auto valueType = input.x.get()->valueType;
    input.x.get()->valueType = ACL_FLOAT16;
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    input.x.get()->valueType = valueType;

    const auto size = input.x.get()->size;
    input.x.get()->size = 3;
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    input.x.get()->size = size;

    const auto base = input.x.get()->idxBase;
    input.x.get()->idxBase = static_cast<aclsparseIndexBase_t>(99);
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    input.x.get()->idxBase = base;

    const auto indexType = input.x.get()->idxType;
    input.x.get()->idxType = static_cast<aclsparseIndexType_t>(99);
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    input.x.get()->idxType = indexType;

    auto unsupportedY = DnVecManager::createConst(4, input.dY.raw(), ACL_INT32);
    auto unsupportedX = SpVecManager::create(4, 2, input.dIndices.get(), input.dX.get(),
                                             ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_INT32);
    EXPECT_EQ(aclsparseGather(input.handle.get(), unsupportedY.cget(), unsupportedX.get()),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(GatherContractTest, RejectsInvalidPointersAndAlias)
{
    GatherContractInputs input;
    input.handle.setStream(env_->stream());
    const auto values = input.x.get()->values;
    input.x.get()->values = input.dY.get();
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    input.x.get()->values = values;

    const auto yValues = input.y.get()->values;
    input.y.get()->values = input.yHost.data();
    EXPECT_EQ(aclsparseGather(input.handle.get(), input.y.cget(), input.x.get()),
              ACL_SPARSE_STATUS_INVALID_VALUE);
    input.y.get()->values = yValues;
}

TEST_F(GatherContractTest, RejectsNonUnitDenseStride)
{
    CheckGatherNonUnitDenseStride(env_->stream());
}
