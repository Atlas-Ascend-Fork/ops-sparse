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
 * @file gather_test.cpp (arch22 / Atlas A2-A3)
 * @brief GTest + CSV-driven test cases for aclsparseGather on Atlas A2/A3.
 *
 * Gather: X.values[i] = Y[X.indices[i] - idxBase] for i = 0 .. nnz-1.
 *
 * arch22 任务范围支持的 dtype：float16 / bfloat16 / float32 / complex64；
 * indices 仅 int32；idxBase 0/1。Host 对其它组合（fp64 / 64I / 非 0-1 base）
 * 显式返回错误，不退回 CPU。
 *
 * 纯数据搬移算子 → 逐位一致（bit-exact）。验证采用 Verifier EXACT 模式：
 *   - f32/f16/bf16：宽化到 float 后逐位比较（宽化是确定性的，等价于 bit-exact）
 *   - c64：实部/虚部展平为 2*nnz 个 float 后逐位比较
 * 对值类型不做数值放宽，确保 gather 一字不差。
 */

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <type_traits>
#include <vector>

#include "acl/acl_base_rt.h"
#include "cann_ops_sparse.h"
#include "fill.h"
#include "test_common.h"
#include "gather_golden.h"
#include "gather_npu_wrapper.h"
#include "gather_param.h"
#include "../../frame/presmoke.h"

#include <algorithm>

using namespace sparse_test;

namespace {

std::vector<GatherTestParam> GetGatherArch22Cases()
{
    auto cases = GetCasesFromCsv<GatherTestParam>("gather_test.csv");
    if (SparseTestFullEnabled()) {
        return cases;
    }

    for (auto& p : cases) {
        // These cases cross a device output block or exercise the largest
        // complex tail. Keep their exact sizes; shrink all other large data
        // movement cases without changing dtype, base or index semantics.
        const bool isBlockBoundaryCase = p.case_name == "float_base0_yblock_multi" ||
                                         p.case_name == "float16_base0_yblock_multi" ||
                                         p.case_name == "c64_base0_ub_worst";
        if (!isBlockBoundaryCase && p.vec_size > 4096) {
            const int oldVecSize = p.vec_size;
            p.vec_size = 4096;
            p.nnz = p.nnz == oldVecSize ? p.vec_size : std::min(p.nnz, 1024);
        }
    }
    return cases;
}

}  // namespace

// 将输出/参考向量转为 float 序列用于 EXACT 验证：
//   - c64：实部/虚部交错展平（2*nnz）
//   - 其它：Verifier::toFloat 宽化（确定性，等价于 bit-exact）
template <typename T>
std::vector<float> ToVerifyVector(const std::vector<T>& v)
{
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::vector<float> out;
        out.reserve(v.size() * 2);
        for (const auto& c : v) {
            out.push_back(c.real());
            out.push_back(c.imag());
        }
        return out;
    } else {
        return Verifier::toFloat(v);
    }
}

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

    std::cout << "==== " << p.case_name << " ==== vec_size=" << p.vec_size << " nnz=" << p.nnz
              << " value_type=" << p.value_type << " idx_base=" << p.idx_base << "\n";

    std::mt19937 rng(p.seed);
    HandleManager handle;
    handle.setStream(stream_);

    VerifyConfig cfg;
    cfg.SetMode(PrecisionMode::EXACT);
    bool pass = false;
    int match_count = 0;

    // c++17 不支持模板 lambda，用 auto 实参 + match_case
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
            pass = Verifier::verifyVector(ToVerifyVector(output), ToVerifyVector(golden), cfg, p.case_name);
        }
    };

    // arch22 任务范围：f16 / bf16 / f32 / c64，indices 仅 I32
    // 注：bf16 在 host 侧以 uint16_t 承载原始位模式（纯搬移，bit-exact 比较等价）
    match_case(ACL_FLOAT, float{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_FLOAT16, __fp16{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_BF16, uint16_t{}, ACL_SPARSE_INDEX_32I, int32_t{});
    match_case(ACL_COMPLEX64, std::complex<float>{}, ACL_SPARSE_INDEX_32I, int32_t{});

    ASSERT_EQ(match_count, 1) << "CSV row dtype/idx_type not matched by any arch22 case";

    EXPECT_TRUE(pass) << "Gather verification failed (" << p.case_name << ")";

    if (pass) {
        std::cout << "[" << p.case_name << "] PASSED\n";
    } else {
        std::cout << "[" << p.case_name << "] FAILED\n";
    }
}

// ===========================================================================
// 非法参数显式报错（不退回 CPU）：fp64 / 64I / 非法 idxBase / valueType 不一致
// ===========================================================================
class GatherInvalidTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

static bool RunGatherExpectFail(aclrtStream stream, aclDataType yType, aclDataType xType, aclsparseIndexType_t idxType,
                                aclsparseIndexBase_t idxBase)
{
    HandleManager handle;
    handle.setStream(stream);

    int64_t N = 64;
    int64_t nnz = 8;
    std::vector<float> yHost(N, 1.0f);
    std::vector<float> xHost(nnz, 0.0f);
    std::vector<int32_t> idxHost(nnz, 0);
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dIdx = DeviceBuffer::copyFrom(idxHost.data(), idxHost.size() * sizeof(int32_t));

    auto dnVecY = DnVecManager::createConst(N, dY.raw(), yType);
    auto spVecX = SpVecManager::create(N, nnz, dIdx.get(), dX.get(), idxType, idxBase, xType);

    aclsparseStatus_t st = aclsparseGather(handle.get(), dnVecY.cget(), spVecX.get());
    return st != ACL_SPARSE_STATUS_SUCCESS;
}

TEST_F(GatherInvalidTest, UnsupportedValueTypeFp64)
{
    // arch22 不支持 fp64
    EXPECT_TRUE(RunGatherExpectFail(env_->stream(), ACL_DOUBLE, ACL_DOUBLE,
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO));
}

TEST_F(GatherInvalidTest, UnsupportedIndexType64I)
{
    // arch22 indices 仅支持 I32
    EXPECT_TRUE(RunGatherExpectFail(env_->stream(), ACL_FLOAT, ACL_FLOAT,
                                    ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_BASE_ZERO));
}

TEST_F(GatherInvalidTest, ValueTypeMismatch)
{
    EXPECT_TRUE(RunGatherExpectFail(env_->stream(), ACL_FLOAT, ACL_FLOAT16,
                                    ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO));
}

TEST_F(GatherInvalidTest, InvalidIdxBase)
{
    // 非法 idxBase（非 0/1，这里用枚举值 2）在稀疏向量描述符创建阶段即被
    // aclsparseCreateSpVec 显式拒绝（返回 INVALID_VALUE），算子不会在非法 base 下执行；
    // 这里直接验证创建阶段即报错（不静默接受，也不退回 CPU）。
    aclrtStream stream = GatherInvalidTest::env_->stream();
    HandleManager handle;
    handle.setStream(stream);
    (void)handle;
    int64_t N = 16;
    int64_t nnz = 4;
    std::vector<float> yHost(N, 1.0f);
    std::vector<float> xHost(nnz, 0.0f);
    std::vector<int32_t> idxHost(nnz, 0);
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dIdx = DeviceBuffer::copyFrom(idxHost.data(), idxHost.size() * sizeof(int32_t));
    (void)dY;
    aclsparseSpVecDescr_t spVecX = nullptr;
    aclsparseStatus_t st = aclsparseCreateSpVec(&spVecX, N, nnz, dIdx.get(), dX.get(),
                                                ACL_SPARSE_INDEX_32I,
                                                static_cast<aclsparseIndexBase_t>(2), ACL_FLOAT);
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE);
    if (spVecX != nullptr) {
        aclsparseDestroySpVec(spVecX);
    }
}

// ===========================================================================
// 重复有效索引 / 越界索引 回归（回应 PR 检视 [重要]）
//   - 重复有效索引合法：y_len 可 < nnz（如 Y=[7], idx=[0,0] base0 → [7,7]），与 torch.index_select 一致
//   - 越界索引（落点不在 [0, y_len)）显式返回 INVALID_VALUE（不静默 clamp）
// ===========================================================================
class GatherRepeatAndRangeTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// 重复有效索引：y_len=1, nnz=2, indices=[0,0]（base0）/ [1,1]（base1）→ 输出 [7,7]
TEST_F(GatherRepeatAndRangeTest, RepeatValidIndex)
{
    aclrtStream stream = env_->stream();
    HandleManager handle;
    handle.setStream(stream);

    std::vector<float> yHost = {7.0f};   // y_len = 1
    const int64_t nnz = 2;
    std::vector<float> golden = {7.0f, 7.0f};

    for (int base : {0, 1}) {
        std::vector<int32_t> idx = (base == 0) ? std::vector<int32_t>{0, 0}
                                               : std::vector<int32_t>{1, 1};
        auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
        auto dIdx = DeviceBuffer::copyFrom(idx.data(), idx.size() * sizeof(int32_t));
        auto dX = DeviceBuffer::alloc(nnz * sizeof(float));

        // 注意：SpVec 的 size 设为 nnz（>= nnz，避免描述符层对 size<nnz 的拒绝）；
        // Gather 正确性只取决于 vecY.nums 与逐索引落点，与 vecX.size 无关。
        auto dnVecY = DnVecManager::createConst(yHost.size(), dY.raw(), ACL_FLOAT);
        auto spVecX = SpVecManager::create(nnz, nnz, dIdx.get(), dX.get(),
                                          ACL_SPARSE_INDEX_32I,
                                          (base == 0) ? ACL_SPARSE_INDEX_BASE_ZERO
                                                      : ACL_SPARSE_INDEX_BASE_ONE,
                                          ACL_FLOAT);

        aclsparseStatus_t st = aclsparseGather(handle.get(), dnVecY.cget(), spVecX.get());
        ASSERT_EQ(st, ACL_SPARSE_STATUS_SUCCESS)
            << "RepeatValidIndex failed (base=" << base << ")";
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

        std::vector<float> out(nnz);
        dX.copyToHost(out.data(), nnz * sizeof(float));
        for (int64_t i = 0; i < nnz; i++) {
            ASSERT_FLOAT_EQ(out[i], golden[i])
                << "RepeatValidIndex mismatch i=" << i << " base=" << base;
        }
    }
}

// 越界索引：落点不在 [0, y_len) 应显式返回 INVALID_VALUE（不静默 clamp）
TEST_F(GatherRepeatAndRangeTest, OutOfRangeIndexReturnsInvalid)
{
    aclrtStream stream = env_->stream();
    HandleManager handle;
    handle.setStream(stream);

    int64_t N = 4;                       // y_len = 4，合法落点 0..3
    int64_t nnz = 3;
    std::vector<float> yHost = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> xHost(nnz, 0.0f);
    std::vector<int32_t> idxHost = {4, 0, 1};   // base0 下 index=4 越界（>= y_len）
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dX = DeviceBuffer::copyFrom(xHost.data(), xHost.size() * sizeof(float));
    auto dIdx = DeviceBuffer::copyFrom(idxHost.data(), idxHost.size() * sizeof(int32_t));

    auto dnVecY = DnVecManager::createConst(N, dY.raw(), ACL_FLOAT);
    auto spVecX = SpVecManager::create(nnz, nnz, dIdx.get(), dX.get(),
                                      ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

    aclsparseStatus_t st = aclsparseGather(handle.get(), dnVecY.cget(), spVecX.get());
    EXPECT_EQ(st, ACL_SPARSE_STATUS_INVALID_VALUE)
        << "OutOfRangeIndex must be rejected (base0, index=4 >= y_len=4)";
}

// ===========================================================================
// handle 校验顺序回归（回应 PR 检视 [建议]）
//   - 空 handle 必须返回 HANDLE_IS_NULLPTR，即使 vecX->nnz == 0（不得被 no-op 早退吞掉）
//   - 零 nnz 的 no-op 仍不要求任何数据指针有效
// ===========================================================================
class GatherHandleOrderTest : public testing::Test {
public:
    static void SetUpTestSuite() { env_ = std::make_unique<AclEnvScope>(); }
    static void TearDownTestSuite() { env_.reset(); }

protected:
    inline static std::unique_ptr<AclEnvScope> env_;
};

// 空 handle + 零 nnz 描述符：handle 校验优先，返回 HANDLE_IS_NULLPTR
TEST_F(GatherHandleOrderTest, NullHandleWithZeroNnzReturnsHandleNullptr)
{
    int64_t N = 4;
    std::vector<float> yHost(N, 1.0f);
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dnVecY = DnVecManager::createConst(N, dY.raw(), ACL_FLOAT);
    // nnz=0 时 indices / values 允许为 nullptr（与 aclsparseCreateSpVec 的约定一致）
    auto spVecX = SpVecManager::create(N, 0, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

    EXPECT_EQ(aclsparseGather(nullptr, dnVecY.cget(), spVecX.get()),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

// 合法 handle + 零 nnz：仍为成功 no-op（同样不要求 indices / values 有效）
TEST_F(GatherHandleOrderTest, ValidHandleZeroNnzReturnsSuccess)
{
    HandleManager handle;
    handle.setStream(env_->stream());

    int64_t N = 4;
    std::vector<float> yHost(N, 1.0f);
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dnVecY = DnVecManager::createConst(N, dY.raw(), ACL_FLOAT);
    auto spVecX = SpVecManager::create(N, 0, nullptr, nullptr, ACL_SPARSE_INDEX_32I,
                                       ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT);

    EXPECT_EQ(aclsparseGather(handle.get(), dnVecY.cget(), spVecX.get()),
              ACL_SPARSE_STATUS_SUCCESS);
}

// 裸指针快路径：零 nnz 的 no-op 不要求任何数据指针；空 handle 仍优先报错
TEST_F(GatherHandleOrderTest, RawZeroNnzNeedsNoDataPointers)
{
    HandleManager handle;
    handle.setStream(env_->stream());

    EXPECT_EQ(aclsparseGatherRaw(handle.get(), nullptr, 0, ACL_FLOAT, nullptr, nullptr, 0,
                                 ACL_SPARSE_INDEX_BASE_ZERO),
              ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseGatherRaw(nullptr, nullptr, 0, ACL_FLOAT, nullptr, nullptr, 0,
                                 ACL_SPARSE_INDEX_BASE_ZERO),
              ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    GatherCases, GatherTest, testing::ValuesIn(GetGatherArch22Cases()),
    [](const testing::TestParamInfo<GatherTestParam>& info) { return info.param.case_name; });
