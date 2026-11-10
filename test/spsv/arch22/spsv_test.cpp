/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "device_buffer.h"
#include "spsv_invalid_enum_calls.h"
#include "spsv_binary64.h"
#include "spsv_tiling_data.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <array>
#include <random>
#include <set>
#include <spawn.h>
#include <sys/wait.h>
#include "spsv_host_utils.h"

namespace {
using Complex = std::complex<double>; // FP64 / complex128 CPU reference.
using spsv_test::CheckAcl;
using spsv_test::CheckSparse;
using spsv_test::DeviceBuffer;
constexpr auto kSuccess = ACL_SPARSE_STATUS_SUCCESS;
constexpr auto kInvalid = ACL_SPARSE_STATUS_INVALID_VALUE;
constexpr auto kDefault = ACL_SPARSE_SPSV_ALG_DEFAULT;
constexpr aclsparseOperation_t kOperations[] = {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
                                                ACL_SPARSE_OP_CONJUGATE_TRANSPOSE};

struct Case {
    aclDataType type = ACL_FLOAT;
    aclsparseFormat_t format = ACL_SPARSE_FORMAT_CSR;
    aclsparseOperation_t operation = ACL_SPARSE_OP_NON_TRANSPOSE;
    bool lower = true;
    bool unit = false;
    int base = 0;
    bool deviceAlpha = false;
    bool inPlace = false;
    std::string name;
};
struct Entry { int row; int col; Complex value; };
struct Matrix {
    int m = 5;
    int width = 0;
    std::vector<int32_t> ptr;
    std::vector<int32_t> idx;
    // One entry per stored value, including SELL padding (row/col == -1).
    std::vector<Entry> entries;
};
Matrix EncodeEntries(int m, const std::vector<Entry> &source, const Case &c);

Complex Value(double real, double imag, const Case &c)
{
    return {real, c.type == ACL_COMPLEX64 ? imag : 0.0};
}
Complex ReferenceDivide(Complex numerator, Complex denominator, aclDataType type)
{
    // Singular fixtures deliberately check IEEE Inf/NaN, not a thrown error.
    const double infinity = std::copysign(std::numeric_limits<double>::infinity(), denominator.real());
    if (type == ACL_FLOAT) {
        return {denominator.real() == 0.0 ? numerator.real() * infinity :
            numerator.real() / denominator.real(), 0.0};
    }
    if (denominator != 0.0) return numerator / denominator;
    return {numerator.real() * infinity, numerator.imag() * infinity};
}
Matrix MakeMatrix(const Case &c)
{
    Matrix a;
    std::vector<Entry> source;
    for (int row = 0; row < a.m; ++row) {
        // Duplicate diagonal entries must be summed, and UNIT must ignore both.
        source.push_back({row, row, Value(3.0 + row * 0.125, 0.25, c)});
        source.push_back({row, row, Value(2.0, -0.125, c)});
        for (int col = 0; col < a.m; ++col) {
            if (row == col) continue;
            const bool selected = c.lower ? col < row : col > row;
            if (selected) {
                const double sign = (row + col) % 2 == 0 ? 1.0 : -1.0;
                source.push_back({row, col, Value(sign * 0.125, 0.0625, c)});
                source.push_back({row, col, Value(-sign * 0.03125, -0.015625, c)});
            } else {
                // The opposite triangle is deliberately nonzero and must be ignored.
                source.push_back({row, col, Value(7.0, 3.0, c)});
            }
        }
    }
    return EncodeEntries(a.m, source, c);
}

std::vector<float> Pack(const std::vector<Complex> &values, aclDataType type)
{
    std::vector<float> out;
    for (const auto &v : values) {
        out.push_back(static_cast<float>(v.real()));
        if (type == ACL_COMPLEX64) out.push_back(static_cast<float>(v.imag()));
    }
    return out;
}
std::vector<Complex> StoredValues(const Matrix &a, const Case &c, bool general)
{
    std::vector<Complex> values;
    for (const auto &e : a.entries) {
        auto v = e.value;
        if (general) v = e.row == e.col && e.row >= 0 ?
            Value(v.real() + 0.5, v.imag() - 0.0625, c) :
            Value(v.real() * 0.5, v.imag() * -0.5, c);
        values.push_back(v);
    }
    return values;
}
std::vector<Complex> Input(const Matrix &a, const Case &c)
{
    std::vector<Complex> x;
    for (int i = 0; i < a.m; ++i) x.push_back(Value((i % 2 ? -1.0 : 1.0) * (1.0 + i * 0.25), 0.125 * (i - 2), c));
    return x;
}
std::vector<Complex> Golden(const Matrix &a, const Case &c, const std::vector<Complex> &values,
    const std::vector<Complex> &x, Complex alpha, const std::vector<Complex> *diagonal = nullptr)
{
    std::vector<Complex> dense(a.m * a.m, Complex{});
    std::vector<uint8_t> present(a.m * a.m, 0);
    // Accumulate duplicates in the encoded storage order, in FP64 / complex128.
    for (size_t p = 0; p < a.entries.size(); ++p) {
        const auto &e = a.entries[p];
        if (e.row < 0 || (c.lower ? e.col > e.row : e.col < e.row)) continue;
        dense[e.row * a.m + e.col] += values[p];
        present[e.row * a.m + e.col] = 1;
    }
    for (int i = 0; i < a.m; ++i) {
        if (diagonal != nullptr) dense[i * a.m + i] = (*diagonal)[i];
        if (c.unit) dense[i * a.m + i] = Complex{1.0, 0.0};
    }
    auto element = [&](int row, int col) {
        auto v = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? dense[row * a.m + col] : dense[col * a.m + row];
        return c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ? std::conj(v) : v;
    };
    const bool forward = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? c.lower : !c.lower;
    std::vector<Complex> y(a.m);
    for (int step = 0; step < a.m; ++step) {
        const int row = forward ? step : a.m - 1 - step;
        Complex rhs = alpha * x[row];
        for (int previous = 0; previous < step; ++previous) {
            const int col = forward ? previous : a.m - 1 - previous;
            const int sourceRow = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? row : col;
            const int sourceCol = c.operation == ACL_SPARSE_OP_NON_TRANSPOSE ? col : row;
            // Structural zeros are not sparse dependencies. Preserve stored
            // zero coefficients so explicit 0 * Inf still propagates NaN.
            if (!present[sourceRow * a.m + sourceCol]) continue;
            rhs -= element(row, col) * y[col];
        }
        // Real singular systems use real IEEE division, avoiding complex 0/0 artifacts.
        y[row] = ReferenceDivide(rhs, element(row, row), c.type);
    }
    return y;
}

void Verify(const std::vector<float> &actual, const std::vector<Complex> &golden, aclDataType type)
{
    const size_t components = type == ACL_COMPLEX64 ? 2 : 1;
    ASSERT_EQ(actual.size(), golden.size() * components);
    for (size_t i = 0; i < golden.size(); ++i) {
        for (size_t component = 0; component < components; ++component) {
            SCOPED_TRACE("element=" + std::to_string(i) + " component=" + std::to_string(component));
            const double expected = component == 0 ? golden[i].real() : golden[i].imag();
            const double got = actual[i * components + component];
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(got));
                continue;
            }
            if (std::isinf(expected)) {
                EXPECT_EQ(got, expected);
                continue;
            }
            ASSERT_TRUE(std::isfinite(got));
            const double error = std::abs(got - expected);
            // SpSV task section 3.2; thresholds and sources are in test/spsv/README.md.
            // All elements must pass (stricter than the task's 99% matching ratio).
            EXPECT_LE(error, std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * std::abs(expected));
            const float rounded = static_cast<float>(expected);
            const double ulp = std::abs(static_cast<double>(std::nextafter(rounded,
                std::numeric_limits<float>::infinity())) - rounded);
            EXPECT_LE(error, std::max(1e-2, 32.0 * ulp));
        }
    }
}

class SpSVArch22Test : public ::testing::Test {
public:
    static void SetUpTestSuite()
    {
        // Logical ACL device 0 maps to the selected physical card in the container.
        int device = 0;
        if (const char *text = std::getenv("SPSV_TEST_DEVICE_ID")) {
            char *end = nullptr;
            const long parsed = std::strtol(text, &end, 10);
            if (*text == '\0' || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max())
                throw std::runtime_error("SPSV_TEST_DEVICE_ID must be a nonnegative logical device ID");
            device = static_cast<int>(parsed);
        }
        env = std::make_unique<sparse_test::AclEnvScope>(device);
        CheckSparse(aclsparseCreate(&handle), "create handle");
        CheckSparse(aclsparseSetStream(handle, env->stream()), "set stream");
    }
    static void TearDownTestSuite()
    {
        if (env) aclrtSynchronizeStream(env->stream());
        if (handle) aclsparseDestroy(handle);
        handle = nullptr;
        env.reset();
    }
    inline static std::unique_ptr<sparse_test::AclEnvScope> env;
    inline static aclsparseHandle_t handle = nullptr;

protected:
    void SetUp() override { ASSERT_EQ(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST), kSuccess); }
};

TEST(SpSVArch22Memory, PublicRangeResultChecksErrorsBoundariesAndOverflow)
{
    // Synthetic addresses are only passed to the pure result checker, never ACL or a kernel.
    const auto *base = reinterpret_cast<const void *>(uintptr_t{4096});
    const auto *inside = reinterpret_cast<const void *>(uintptr_t{4100});
    const auto *before = reinterpret_cast<const void *>(uintptr_t{4092});
    const auto *end = reinterpret_cast<const void *>(uintptr_t{8192});
    EXPECT_EQ(spsv_memory::ValidateRangeResult(base, 4096, ACL_SUCCESS, base, 4096), ACL_SUCCESS);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(inside, 4092, ACL_SUCCESS, base, 4096), ACL_SUCCESS);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(inside, 0, ACL_SUCCESS, base, 4096), ACL_SUCCESS);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(inside, 4093, ACL_SUCCESS, base, 4096),
        ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(before, 4, ACL_SUCCESS, base, 4096), ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(end, 0, ACL_SUCCESS, base, 4096), ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(base, 4, ACL_SUCCESS, base, 0), ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(base, 4, ACL_SUCCESS, nullptr, 4096), ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(nullptr, 4, ACL_SUCCESS, base, 4096), ACL_ERROR_RT_PARAM_INVALID);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(base, 4, ACL_ERROR_RT_FEATURE_NOT_SUPPORT, nullptr, 0),
        ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(base, 4, ACL_ERROR_RT_PARAM_INVALID, base, 4096),
        ACL_ERROR_RT_PARAM_INVALID);
    const auto *nearLimit = reinterpret_cast<const void *>(std::numeric_limits<uintptr_t>::max() - 3);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(nearLimit, 4, ACL_SUCCESS, nearLimit, 4), ACL_SUCCESS);
    EXPECT_EQ(spsv_memory::ValidateRangeResult(nearLimit, 4, ACL_SUCCESS, nearLimit, 8), ACL_ERROR_RT_PARAM_INVALID);
    for (const auto result : {ACL_SUCCESS, ACL_ERROR_RT_FEATURE_NOT_SUPPORT}) {
        EXPECT_EQ(spsv_memory::ValidateRangeResult(nearLimit, 8, result, nearLimit, 8), ACL_ERROR_RT_PARAM_INVALID);
        EXPECT_EQ(spsv_memory::ValidateRangeResult(nullptr, 0, result, base, 4096), ACL_ERROR_RT_PARAM_INVALID);
    }
}

TEST_F(SpSVArch22Test, PublicRangeChecksLiveAllocationAndOffsets)
{
    DeviceBuffer buffer(4096);
    void *pointer = buffer.get();
    void *base = nullptr;
    size_t bytes = 0;
    const auto result = aclrtMemGetAddressRange(pointer, &base, &bytes);
    if (result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        EXPECT_EQ(spsv_memory::ValidateRange(pointer, 4096), ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
        return;
    }
    ASSERT_EQ(result, ACL_SUCCESS);
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    const auto start = reinterpret_cast<uintptr_t>(base);
    ASSERT_TRUE(spsv_memory::Contains(address, 4096, start, bytes));
    EXPECT_EQ(spsv_memory::ValidateRange(pointer, 4096), ACL_SUCCESS);
    EXPECT_EQ(spsv_memory::ValidateRange(reinterpret_cast<void *>(address + 4), 4092), ACL_SUCCESS);
    const size_t remaining = bytes - (address - start);
    EXPECT_EQ(spsv_memory::ValidateRange(pointer, remaining), ACL_SUCCESS);
    ASSERT_LT(remaining, std::numeric_limits<size_t>::max());
    // Only query the live pointer; the oversized span is never submitted to a kernel.
    EXPECT_EQ(spsv_memory::ValidateRange(pointer, remaining + 1), ACL_ERROR_RT_PARAM_INVALID);
}

class Plan {
public:
    Plan(const Matrix &matrix, const Case &config) : a(matrix), c(config),
        initial(Pack(StoredValues(a, c, false), c.type)), input(Pack(Input(a, c), c.type)),
        dPtr(a.ptr.size() * sizeof(int32_t)), dIdx(a.idx.size() * sizeof(int32_t)),
        dValues(initial.size() * sizeof(float)), dX(input.size() * sizeof(float)),
        dY(input.size() * sizeof(float)), dAlpha((c.type == ACL_COMPLEX64 ? 2 : 1) * sizeof(float))
    {
        try {
            dPtr.Put(a.ptr); dIdx.Put(a.idx); dValues.Put(initial); dX.Put(input);
            alpha = Pack({Value(0.75, -0.25, c)}, c.type);
            dAlpha.Put(alpha);
            auto base = c.base == 0 ? ACL_SPARSE_INDEX_BASE_ZERO : ACL_SPARSE_INDEX_BASE_ONE;
            aclsparseStatus_t status;
            if (c.format == ACL_SPARSE_FORMAT_CSR) status = aclsparseCreateCsr(&mat, a.m, a.m, a.entries.size(),
                dPtr.get(), dIdx.get(), dValues.get(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, c.type);
            else if (c.format == ACL_SPARSE_FORMAT_CSC) status = aclsparseCreateCsc(&mat, a.m, a.m, a.entries.size(),
                dPtr.get(), dIdx.get(), dValues.get(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, base, c.type);
            else if (c.format == ACL_SPARSE_FORMAT_COO) status = aclsparseCreateCoo(&mat, a.m, a.m, a.entries.size(),
                dPtr.get(), dIdx.get(), dValues.get(), ACL_SPARSE_INDEX_32I, base, c.type);
            else status = aclsparseCreateSlicedEll(&mat, a.m, a.m, a.entries.size(), a.width, a.entries.empty() ? 0 : 1,
                dPtr.get(), dIdx.get(), dValues.get(), ACL_SPARSE_INDEX_32I, base, c.type);
            CheckSparse(status, "create matrix");
            auto fill = c.lower ? ACL_SPARSE_FILL_MODE_LOWER : ACL_SPARSE_FILL_MODE_UPPER;
            auto diag = c.unit ? ACL_SPARSE_DIAG_TYPE_UNIT : ACL_SPARSE_DIAG_TYPE_NON_UNIT;
            CheckSparse(aclsparseSpMatSetAttribute(mat, ACL_SPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)), "fill");
            CheckSparse(aclsparseSpMatSetAttribute(mat, ACL_SPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)), "diag");
            CheckSparse(aclsparseCreateDnVec(&x, a.m, dX.get(), c.type), "create X");
            CheckSparse(aclsparseCreateDnVec(&y, a.m, c.inPlace ? dX.get() : dY.get(), c.type), "create Y");
            CheckSparse(aclsparseSpSV_createDescr(&state), "create plan");
            CheckSparse(aclsparseSetPointerMode(SpSVArch22Test::handle, c.deviceAlpha ?
                ACL_SPARSE_POINTER_MODE_DEVICE : ACL_SPARSE_POINTER_MODE_HOST), "pointer mode");
        } catch (...) {
            ReleaseDescriptors();
            throw;
        }
    }
    ~Plan()
    {
        // Complete asynchronous solve/update before descriptors or storage are released.
        if (SpSVArch22Test::env) aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ReleaseDescriptors();
    }
    void ReleaseDescriptors()
    {
        if (state) aclsparseSpSV_destroyDescr(state);
        if (y) aclsparseDestroyDnVec(y);
        if (x) aclsparseDestroyDnVec(x);
        if (mat) aclsparseDestroySpMat(mat);
        state = nullptr;
        x = nullptr;
        y = nullptr;
        mat = nullptr;
    }
    Plan(const Plan &) = delete;
    Plan &operator=(const Plan &) = delete;
    const void *Alpha() const { return c.deviceAlpha ? dAlpha.get() : alpha.data(); }
    Complex AlphaValue() const { return {alpha[0], c.type == ACL_COMPLEX64 ? alpha[1] : 0.0}; }
    aclsparseStatus_t Size(size_t *bytes, aclsparseConstDnVecDescr_t vx = nullptr, aclsparseDnVecDescr_t vy = nullptr,
        aclsparseOperation_t op = ACL_SPARSE_OP_NON_TRANSPOSE, aclDataType type = ACL_FLOAT,
        aclsparseSpSVAlg_t algorithm = kDefault)
    {
        return aclsparseSpSV_bufferSize(SpSVArch22Test::handle, op, Alpha(), mat, vx, vy, type, algorithm, state,
                                        bytes);
    }
    aclsparseStatus_t Analyze(bool bindVectors = false)
    {
        size_t bytes = 0;
        CheckSparse(Size(&bytes, bindVectors ? x : nullptr, bindVectors ? y : nullptr, c.operation, c.type),
                    "bufferSize");
        workspace = std::make_unique<DeviceBuffer>(bytes);
        return aclsparseSpSV_analysis(SpSVArch22Test::handle, c.operation, Alpha(), mat,
            bindVectors ? x : nullptr, bindVectors ? y : nullptr, c.type, kDefault, state, workspace->get());
    }
    aclsparseStatus_t Solve(aclsparseOperation_t op)
    {
        return aclsparseSpSV_solve(SpSVArch22Test::handle, op, Alpha(), mat, x, y, c.type, kDefault, state);
    }
    void VerifyTwice(const std::vector<Complex> &expected)
    {
        std::vector<float> first;
        for (int repeat = 0; repeat < 2; ++repeat) {
            // An in-place result must not become the RHS of the next repeat.
            dX.Put(input);
            const auto status = Solve(c.operation);
            const auto sync = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
            ASSERT_EQ(status, kSuccess);
            ASSERT_EQ(sync, ACL_SUCCESS);
            auto actual = (c.inPlace ? dX : dY).Get<float>(input.size());
            Verify(actual, expected, c.type);
            if (repeat == 0) first = actual;
            else EXPECT_EQ(std::memcmp(first.data(), actual.data(), actual.size() * sizeof(float)), 0);
            if (!c.inPlace) EXPECT_EQ(dX.Get<float>(input.size()), input);
        }
        EXPECT_EQ(dPtr.Get<int32_t>(a.ptr.size()), a.ptr);
        EXPECT_EQ(dIdx.Get<int32_t>(a.idx.size()), a.idx);
        EXPECT_EQ(dValues.Get<float>(initial.size()), initial);
        EXPECT_EQ(dAlpha.Get<float>(alpha.size()), alpha);
    }
    Matrix a;
    Case c;
    std::vector<float> initial, input, alpha;
    DeviceBuffer dPtr, dIdx, dValues, dX, dY, dAlpha;
    aclsparseSpMatDescr_t mat = nullptr;
    aclsparseDnVecDescr_t x = nullptr, y = nullptr;
    aclsparseSpSVDescr_t state = nullptr;
    std::unique_ptr<DeviceBuffer> workspace;
};

class UpdateInput {
public:
    UpdateInput(const std::vector<Complex> &values, aclDataType type)
        : packed_(Pack(values, type)), device_(packed_.size() * sizeof(float)) { device_.Put(packed_); }
    aclsparseStatus_t Apply(Plan &plan, aclsparseSpSVUpdate_t part)
    {
        return aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, device_.get(), part);
    }
    void ExpectUnchanged() const { EXPECT_EQ(device_.Get<float>(packed_.size()), packed_); }
private:
    std::vector<float> packed_;
    DeviceBuffer device_;
};

struct GuardedVectors {
    static constexpr size_t kGuard = 32;
    explicit GuardedVectors(const Plan &plan)
        : x((plan.input.size() + 2 * kGuard) * sizeof(float)),
          y((plan.input.size() + 2 * kGuard) * sizeof(float)) {}
    aclsparseStatus_t Bind(Plan &plan)
    {
        auto *xValues = static_cast<float *>(x.get()) + kGuard;
        auto *yValues = plan.c.inPlace ? xValues : static_cast<float *>(y.get()) + kGuard;
        const auto status = aclsparseDnVecSetValues(plan.x, xValues);
        return status == kSuccess ? aclsparseDnVecSetValues(plan.y, yValues) : status;
    }
    DeviceBuffer x, y;
};

class SpSVArch22Functional : public SpSVArch22Test, public ::testing::WithParamInterface<Case> {};
TEST_P(SpSVArch22Functional, SolveGeneralDiagonalAndDeterminism)
{
    const auto c = GetParam();
    Plan plan(MakeMatrix(c), c);
    // NULL vector descriptors are explicitly allowed by bufferSize/analysis.
    ASSERT_EQ(plan.Analyze(), kSuccess);
    const auto x = Input(plan.a, c);
    const auto initial = StoredValues(plan.a, c, false);
    plan.VerifyTwice(Golden(plan.a, c, initial, x, plan.AlphaValue()));
    if (::testing::Test::HasFatalFailure()) return;

    const auto general = StoredValues(plan.a, c, true);
    UpdateInput dGeneral(general, c.type);
    ASSERT_EQ(dGeneral.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    plan.VerifyTwice(Golden(plan.a, c, general, x, plan.AlphaValue()));
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<Complex> diagonal;
    for (int row = 0; row < plan.a.m; ++row) diagonal.push_back(Value(8.0 + row * 0.25, -0.375, c));
    UpdateInput dDiagonal(diagonal, c.type);
    ASSERT_EQ(dDiagonal.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    // Replace the logical diagonal once, even when storage contains duplicates.
    plan.VerifyTwice(Golden(plan.a, c, general, x, plan.AlphaValue(), &diagonal));
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    dGeneral.ExpectUnchanged();
    dDiagonal.ExpectUnchanged();
}

struct AsyncUpdateSources {
    std::array<std::vector<Complex>, 2> general, diagonal;
    std::array<std::vector<float>, 2> packedGeneral, packedDiagonal;
    std::array<std::unique_ptr<DeviceBuffer>, 2> sourceGeneral, sourceDiagonal;
};
void InitializeAsyncSources(const Case &c, const Matrix &a, AsyncUpdateSources &sources)
{
    auto &general = sources.general;
    auto &diagonal = sources.diagonal;
    auto &packedGeneral = sources.packedGeneral;
    auto &packedDiagonal = sources.packedDiagonal;
    auto &sourceGeneral = sources.sourceGeneral;
    auto &sourceDiagonal = sources.sourceDiagonal;
    for (size_t variant = 0; variant < 2; ++variant) {
        general[variant] = StoredValues(a, c, true);
        if (variant != 0) for (auto &v : general[variant]) v *= 0.5;
        for (int row = 0; row < a.m; ++row)
            diagonal[variant].push_back(Value(8.0 + row * 0.25 + variant, -0.375, c));
        packedGeneral[variant] = Pack(general[variant], c.type);
        packedDiagonal[variant] = Pack(diagonal[variant], c.type);
        sourceGeneral[variant] = std::make_unique<DeviceBuffer>(packedGeneral[variant].size() * sizeof(float));
        sourceDiagonal[variant] = std::make_unique<DeviceBuffer>(packedDiagonal[variant].size() * sizeof(float));
        sourceGeneral[variant]->Put(packedGeneral[variant]);
        sourceDiagonal[variant]->Put(packedDiagonal[variant]);
    }
}
TEST_P(SpSVArch22Functional, SameStreamAsyncProducerBeforeGeneralAndDiagonal)
{
    const auto c = GetParam();
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto x = Input(plan.a, c);
    AsyncUpdateSources sources;
    InitializeAsyncSources(c, plan.a, sources);
    const auto &general = sources.general;
    const auto &diagonal = sources.diagonal;
    const auto &packedGeneral = sources.packedGeneral;
    const auto &packedDiagonal = sources.packedDiagonal;
    const auto &sourceGeneral = sources.sourceGeneral;
    const auto &sourceDiagonal = sources.sourceDiagonal;
    DeviceBuffer originalX(plan.input.size() * sizeof(float));
    originalX.Put(plan.input);
    DeviceBuffer updateGeneral(packedGeneral[0].size() * sizeof(float));
    DeviceBuffer updateDiagonal(packedDiagonal[0].size() * sizeof(float));
    // Inputs are distinct from matA.values. Do not hide an update bug by
    // mutating that original allocation or by using a framework queue drain.
    for (int repeat = 0; repeat < 8; ++repeat) {
        const size_t variant = static_cast<size_t>(repeat % 2);
        for (bool diagOnly : {false, true}) {
            SCOPED_TRACE("repeat=" + std::to_string(repeat) + (diagOnly ? " DIAGONAL" : " GENERAL"));
            auto &destination = diagOnly ? updateDiagonal : updateGeneral;
            const auto &source = diagOnly ? sourceDiagonal[variant] : sourceGeneral[variant];
            const size_t bytes = (diagOnly ? packedDiagonal[variant].size() : packedGeneral[variant].size()) *
                                 sizeof(float);
            // These producers, update, and solve are submitted to the same
            // ACL stream without an intervening stream/device synchronization.
            const auto producedX = aclrtMemcpyAsync(plan.dX.get(), plan.input.size() * sizeof(float),
                originalX.get(), plan.input.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE, env->stream());
            const auto producedValues = aclrtMemcpyAsync(destination.get(), bytes, source->get(), bytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, env->stream());
            const auto part = diagOnly ? ACL_SPARSE_SPSV_UPDATE_DIAGONAL : ACL_SPARSE_SPSV_UPDATE_GENERAL;
            const auto updated = aclsparseSpSV_updateMatrix(handle, plan.state, destination.get(), part);
            const auto solved = plan.Solve(c.operation);
            const auto synchronized = aclrtSynchronizeStream(env->stream());
            ASSERT_EQ(producedX, ACL_SUCCESS); ASSERT_EQ(producedValues, ACL_SUCCESS);
            ASSERT_EQ(updated, kSuccess); ASSERT_EQ(solved, kSuccess);
            ASSERT_EQ(synchronized, ACL_SUCCESS);
            const auto actual = (c.inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
            Verify(actual, Golden(plan.a, c, general[variant], x, plan.AlphaValue(),
                diagOnly ? &diagonal[variant] : nullptr), c.type);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
    EXPECT_EQ(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    for (size_t variant = 0; variant < 2; ++variant) {
        EXPECT_EQ(sourceGeneral[variant]->Get<float>(packedGeneral[variant].size()), packedGeneral[variant]);
        EXPECT_EQ(sourceDiagonal[variant]->Get<float>(packedDiagonal[variant].size()), packedDiagonal[variant]);
    }
}

std::vector<Case> FunctionalCases()
{
    std::vector<Case> cases;
    const std::vector<aclsparseFormat_t> formats = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC,
        ACL_SPARSE_FORMAT_COO, ACL_SPARSE_FORMAT_SLICED_ELL};
    const char *names[] = {"CSR", "CSC", "COO", "SELL"};
    for (int type = 0; type < 2; ++type) for (size_t format = 0; format < formats.size(); ++format)
        for (int op = 0; op < 3; ++op) for (int lower = 0; lower < 2; ++lower)
            for (int unit = 0; unit < 2; ++unit) for (int base = 0; base < 2; ++base) {
                Case c;
                c.type = type ? ACL_COMPLEX64 : ACL_FLOAT;
                c.format = formats[format];
                c.operation = static_cast<aclsparseOperation_t>(op);
                c.lower = lower != 0;
                c.unit = unit != 0;
                c.base = base;
                // Each datatype/format exercises both pointer modes and alias modes.
                c.deviceAlpha = (base + op + lower) % 2 != 0;
                c.inPlace = (unit + op + static_cast<int>(format)) % 2 != 0;
                c.name = std::string(type ? "C64_" : "F32_") + names[format] + "_" + "NTH"[op] +
                    (c.lower ? "_Lower" : "_Upper") + (c.unit ? "_Unit" : "_Nonunit") +
                    "_Base" + std::to_string(base) + (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") +
                    (c.inPlace ? "_InPlace" : "_Separate");
                cases.push_back(c);
            }
    return cases;
}
INSTANTIATE_TEST_SUITE_P(CoreMatrix, SpSVArch22Functional, ::testing::ValuesIn(FunctionalCases()),
    [](const ::testing::TestParamInfo<Case> &info) { return info.param.name; });

TEST_F(SpSVArch22Test, EmptyMatrixHasZeroWorkspaceAndAcceptsNullStorage)
{
    Matrix empty;
    empty.m = 0;
    Case c;
    Plan plan(empty, c);
    size_t bytes = 123;
    ASSERT_EQ(plan.Size(&bytes), kSuccess);
    EXPECT_EQ(bytes, 0u);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kSuccess);
}

TEST_F(SpSVArch22Test, ZeroNnzUnitAndNonunitUseIeeeRealSemantics)
{
    for (bool unit : {false, true}) {
        SCOPED_TRACE(unit ? "unit" : "nonunit");
        Matrix empty; empty.m = 3; empty.ptr = {0, 0, 0, 0};
        Case c; c.unit = unit;
        Plan plan(empty, c);
        ASSERT_EQ(plan.Analyze(), kSuccess);
        plan.VerifyTwice(Golden(empty, c, {}, Input(empty, c), plan.AlphaValue()));
    }
}

TEST_F(SpSVArch22Test, OneElementComplexConjugateDiagonal)
{
    Case c;
    c.type = ACL_COMPLEX64;
    c.operation = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    c.deviceAlpha = true;
    Matrix a;
    a.m = 1;
    a.ptr = {0, 1};
    a.idx = {0};
    a.entries = {{0, 0, {2.0, 1.0}}};
    Plan plan(a, c);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    plan.VerifyTwice(Golden(a, c, {{2.0, 1.0}}, Input(a, c), plan.AlphaValue()));
}

TEST_F(SpSVArch22Test, MissingAndZeroRealDiagonalPropagateSignedInfinity)
{
    for (bool stored : {false, true}) {
        Case c; c.base = 1;
        Matrix a; a.m = 1; a.ptr = stored ? std::vector<int32_t>{1, 2} : std::vector<int32_t>{1, 1};
        if (stored) {
            a.idx = {1};
            a.entries = {{0, 0, {0.0, 0.0}}};
        }
        Plan plan(a, c);
        ASSERT_EQ(plan.Analyze(), kSuccess);
        // Do not substitute a benign diagonal into the singular Golden.
        plan.VerifyTwice(Golden(a, c, StoredValues(a, c, false), Input(a, c), plan.AlphaValue()));
    }
}

TEST_F(SpSVArch22Test, ExplicitZeroOffDiagonalPropagatesInfinityToNan)
{
    Case c;
    Matrix a;
    a.m = 2;
    a.ptr = {0, 0, 2};
    a.idx = {0, 1};
    a.entries = {{1, 0, {0.0, 0.0}}, {1, 1, {2.0, 0.0}}};
    Plan plan(a, c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    // Row 0 has a missing diagonal and yields +Inf. Row 1 has an explicit
    // zero dependency on row 0: its stored 0 * Inf must produce NaN.
    const auto expected = Golden(a, c, StoredValues(a, c, false), Input(a, c), plan.AlphaValue());
    ASSERT_TRUE(std::isinf(expected[0].real()));
    ASSERT_GT(expected[0].real(), 0.0);
    ASSERT_TRUE(std::isnan(expected[1].real()));
    plan.VerifyTwice(expected);
}

TEST_F(SpSVArch22Test, AnalysisWithoutVectorsAllowsNewSolveDescriptorsAndStorage)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    plan.VerifyTwice(Golden(plan.a, c, StoredValues(plan.a, c, false), Input(plan.a, c), plan.AlphaValue()));
    if (::testing::Test::HasFatalFailure()) return;
    // NULL-vector Analysis permits fresh vector descriptors for each Solve.
    // An analysis with NULL vectors therefore binds only the matrix and plan.
    DeviceBuffer nextX(plan.input.size() * sizeof(float));
    DeviceBuffer nextY(plan.input.size() * sizeof(float));
    auto changed = Input(plan.a, c);
    for (auto &v : changed) v *= -0.5;
    const auto packed = Pack(changed, c.type);
    nextX.Put(packed);
    ASSERT_EQ(aclsparseDestroyDnVec(plan.x), kSuccess);
    plan.x = nullptr;
    ASSERT_EQ(aclsparseDestroyDnVec(plan.y), kSuccess);
    plan.y = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&plan.x, plan.a.m, nextX.get(), c.type), kSuccess);
    ASSERT_EQ(aclsparseCreateDnVec(&plan.y, plan.a.m, nextY.get(), c.type), kSuccess);
    const auto status = plan.Solve(c.operation);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    ASSERT_EQ(status, kSuccess);
    Verify(nextY.Get<float>(packed.size()),
           Golden(plan.a, c, StoredValues(plan.a, c, false), changed, plan.AlphaValue()), c.type);
    EXPECT_EQ(nextX.Get<float>(packed.size()), packed);
}

TEST_F(SpSVArch22Test, DenseDimensionTypeStrideAndHostStorageAreRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    ASSERT_EQ(aclsparseDnVecSetStride(plan.x, 2), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(aclsparseDnVecSetStride(plan.x, 1), kSuccess);
    // Host storage is live, aligned, and contains enough elements for the view.
    ASSERT_EQ(aclsparseDnVecSetValues(plan.x, plan.input.data()), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(aclsparseDnVecSetValues(plan.x, plan.dX.get()), kSuccess);
    ASSERT_EQ(aclsparseDestroyDnVec(plan.y), kSuccess);
    plan.y = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&plan.y, plan.a.m - 1, plan.dY.get(), c.type), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(aclsparseDestroyDnVec(plan.y), kSuccess);
    plan.y = nullptr;
    ASSERT_EQ(aclsparseCreateDnVec(&plan.y, plan.a.m, plan.dY.get(), ACL_COMPLEX64), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
}

TEST_F(SpSVArch22Test, InvalidOperationTypeAlgorithmAndNullParameters)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    size_t bytes = 0;
    EXPECT_EQ(SpsvTestBufferSizeEnums(handle, 99, plan.Alpha(), plan.mat, c.type, kDefault, plan.state, &bytes),
              kInvalid);
    EXPECT_EQ(plan.Size(&bytes, nullptr, nullptr, c.operation, ACL_DOUBLE), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(SpsvTestBufferSizeEnums(handle, c.operation, plan.Alpha(), plan.mat, c.type, 99, plan.state, &bytes),
              ACL_SPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(plan.Size(nullptr), kInvalid);
    EXPECT_EQ(aclsparseSpSV_createDescr(nullptr), kInvalid);
    EXPECT_EQ(aclsparseSpSV_bufferSize(nullptr, c.operation, plan.Alpha(), plan.mat, nullptr, nullptr,
        c.type, kDefault, plan.state, &bytes), ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclsparseSpSV_bufferSize(handle, c.operation, nullptr, plan.mat, nullptr, nullptr,
        c.type, kDefault, plan.state, &bytes), kInvalid);
    EXPECT_EQ(aclsparseSpSV_updateMatrix(handle, plan.state, plan.dValues.get(), ACL_SPARSE_SPSV_UPDATE_GENERAL),
              kInvalid);
}

TEST_F(SpSVArch22Test, Arch22RejectsI64IndicesBeforeReadingStorage)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(aclsparseDestroySpMat(plan.mat), kSuccess);
    plan.mat = nullptr;
    // Common descriptor construction allows I64. arch22 SpSV must reject it
    // before interpreting these buffers; no I64 device kernel is launched.
    ASSERT_EQ(aclsparseCreateCsr(&plan.mat, plan.a.m, plan.a.m, plan.a.entries.size(), plan.dPtr.get(),
        plan.dIdx.get(), plan.dValues.get(), ACL_SPARSE_INDEX_64I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, ACL_FLOAT), kSuccess);
    size_t bytes = 0;
    EXPECT_EQ(plan.Size(&bytes), ACL_SPARSE_STATUS_NOT_SUPPORTED);
}

TEST_F(SpSVArch22Test, SolveBeforeAnalysisAndNullSolveVectorsAreRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    EXPECT_EQ(aclsparseSpSV_solve(handle, c.operation, plan.Alpha(), plan.mat, nullptr, plan.y,
        c.type, kDefault, plan.state), kInvalid);
    EXPECT_EQ(aclsparseSpSV_solve(handle, c.operation, plan.Alpha(), plan.mat, plan.x, nullptr,
        c.type, kDefault, plan.state), kInvalid);
    EXPECT_EQ(aclsparseSpSV_updateMatrix(handle, plan.state, nullptr, ACL_SPARSE_SPSV_UPDATE_GENERAL), kInvalid);
    EXPECT_EQ(SpsvTestUpdateEnum(handle, plan.state, plan.dValues.get(), 99), kInvalid);
}

TEST_F(SpSVArch22Test, BindingChangesRequireReanalysis)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    EXPECT_EQ(plan.Solve(ACL_SPARSE_OP_TRANSPOSE), kInvalid);
    auto upper = ACL_SPARSE_FILL_MODE_UPPER;
    ASSERT_EQ(aclsparseSpMatSetAttribute(plan.mat, ACL_SPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    auto lower = ACL_SPARSE_FILL_MODE_LOWER;
    ASSERT_EQ(aclsparseSpMatSetAttribute(plan.mat, ACL_SPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)), kSuccess);
    ASSERT_EQ(aclsparseDnVecSetValues(plan.x, plan.dY.get()), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(aclsparseDnVecSetValues(plan.x, plan.dX.get()), kSuccess);
    ASSERT_EQ(aclsparseSpMatSetValues(plan.mat, plan.dY.get()), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    ASSERT_EQ(aclsparseSpMatSetValues(plan.mat, plan.dValues.get()), kSuccess);
    ASSERT_EQ(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_DEVICE), kSuccess);
    EXPECT_EQ(aclsparseSpSV_solve(handle, c.operation, plan.dAlpha.get(), plan.mat, plan.x, plan.y,
        c.type, kDefault, plan.state), kInvalid);
    ASSERT_EQ(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_HOST), kSuccess);
    // A different stream is a different binding; neither stream has pending work.
    aclrtStream other = nullptr;
    ASSERT_EQ(aclrtCreateStream(&other), ACL_SUCCESS);
    const auto setOther = aclsparseSetStream(handle, other);
    EXPECT_EQ(setOther, kSuccess);
    if (setOther == kSuccess) EXPECT_EQ(plan.Solve(c.operation), kInvalid);
    EXPECT_EQ(aclsparseSetStream(handle, env->stream()), kSuccess);
    EXPECT_EQ(aclrtDestroyStream(other), ACL_SUCCESS);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    plan.VerifyTwice(Golden(plan.a, c, StoredValues(plan.a, c, false), Input(plan.a, c), plan.AlphaValue()));
}

TEST_F(SpSVArch22Test, InvalidWorkspaceAndHostAlphaInDeviceModeAreRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    size_t bytes = 0;
    ASSERT_EQ(plan.Size(&bytes), kSuccess);
    ASSERT_GT(bytes, 0u);
    EXPECT_EQ(aclsparseSpSV_analysis(handle, c.operation, plan.Alpha(), plan.mat, nullptr, nullptr,
        c.type, kDefault, plan.state, nullptr), kInvalid);
    DeviceBuffer workspace(bytes + 512);
    EXPECT_EQ(aclsparseSpSV_analysis(handle, c.operation, plan.Alpha(), plan.mat, nullptr, nullptr,
        c.type, kDefault, plan.state, static_cast<char *>(workspace.get()) + 4), kInvalid);
    ASSERT_EQ(aclsparseSetPointerMode(handle, ACL_SPARSE_POINTER_MODE_DEVICE), kSuccess);
    EXPECT_EQ(plan.Size(&bytes), kInvalid);
}

TEST_F(SpSVArch22Test, SafePartialOverlapIsRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    // Both views are within a live allocation, so no invalid/free pointer is tested.
    DeviceBuffer shared((plan.a.m + 1) * sizeof(float));
    ASSERT_EQ(aclsparseDnVecSetValues(plan.x, shared.get()), kSuccess);
    ASSERT_EQ(aclsparseDnVecSetValues(plan.y, static_cast<float *>(shared.get()) + 1), kSuccess);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    EXPECT_EQ(plan.Solve(c.operation), kInvalid);
}

TEST_F(SpSVArch22Test, MalformedStructuresAreRejectedForAllFormats)
{
    for (auto format : {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC, ACL_SPARSE_FORMAT_COO,
        ACL_SPARSE_FORMAT_SLICED_ELL}) {
        Case c; c.format = format;
        auto a = MakeMatrix(c);
        // Out-of-range coordinate is in-bounds storage and is rejected in analysis.
        a.idx[0] = a.m;
        Plan plan(a, c);
        EXPECT_EQ(plan.Analyze(), kInvalid) << "format=" << static_cast<int>(format);
    }
    for (int kind = 0; kind < 3; ++kind) {
        Case c;
        auto a = MakeMatrix(c);
        if (kind == 0) a.ptr[0] = 1;
        if (kind == 1) a.ptr[2] = a.ptr[1] - 1;
        if (kind == 2) --a.ptr.back();
        Plan plan(a, c);
        EXPECT_EQ(plan.Analyze(), kInvalid) << "CSR pointer error=" << kind;
    }
    Case sell;
    sell.format = ACL_SPARSE_FORMAT_SLICED_ELL;
    auto a = MakeMatrix(sell);
    --a.width; // Pointer span no longer equals m * sliceNnz.
    Plan plan(a, sell);
    EXPECT_EQ(plan.Analyze(), kInvalid);
}

struct DiagonalCase {
    Case config;
    int rows;
};

Matrix MakeDuplicateDiagonal(const DiagonalCase &test)
{
    Matrix a;
    a.m = test.rows;
    a.ptr.push_back(test.config.base);
    for (int row = 0; row < a.m; ++row) {
        // Two stored entries for each diagonal exercise deterministic merging
        // and DIAGONAL replacement of the logical coefficient exactly once.
        a.entries.push_back({row, row, Value(2.0, -0.0625, test.config)});
        a.entries.push_back({row, row, Value(4.0 + (row % 17) * 0.03125, 0.25, test.config)});
        a.idx.push_back(row + test.config.base);
        a.idx.push_back(row + test.config.base);
        a.ptr.push_back(test.config.base + static_cast<int32_t>(a.entries.size()));
    }
    return a;
}

std::vector<Complex> DiagonalGolden(const Matrix &a, const Case &c,
    const std::vector<Complex> &values, const std::vector<Complex> &x, Complex alpha,
    const std::vector<Complex> *replacement = nullptr)
{
    // O(m + nnz) storage/work: no m*m dense allocation for the multicore case.
    std::vector<Complex> diagonal(a.m, Complex{});
    for (size_t p = 0; p < a.entries.size(); ++p) {
        const auto &entry = a.entries[p];
        diagonal[entry.row] += values[p];
    }
    if (replacement != nullptr) diagonal = *replacement;
    std::vector<Complex> y(a.m);
    for (int row = 0; row < a.m; ++row) {
        auto d = c.unit ? Complex{1.0, 0.0} : diagonal[row];
        if (c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) d = std::conj(d);
        const auto rhs = alpha * x[row];
        y[row] = ReferenceDivide(rhs, d, c.type);
    }
    return y;
}

void VerifyGuardedDiagonalTwice(Plan &plan, GuardedVectors &vectors, const std::vector<Complex> &expected)
{
    auto &guardedX = vectors.x;
    auto &guardedY = vectors.y;
    constexpr size_t guard = 32; // A 128-byte prefix keeps the values 32-byte aligned.
    constexpr float canary = -123456.75f;
    const size_t count = plan.input.size();
    const std::vector<float> expectedGuard(guard, canary);
    std::vector<float> inputWithGuards(count + 2 * guard, canary);
    std::copy(plan.input.begin(), plan.input.end(), inputWithGuards.begin() + guard);
    const std::vector<float> blankOutput(count + 2 * guard, canary);
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        guardedX.Put(inputWithGuards);
        guardedY.Put(blankOutput);
        const auto status = plan.Solve(plan.c.operation);
        const auto sync = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(status, kSuccess);
        ASSERT_EQ(sync, ACL_SUCCESS);
        const auto output = (plan.c.inPlace ? guardedX : guardedY).Get<float>(count + 2 * guard);
        EXPECT_EQ(std::vector<float>(output.begin(), output.begin() + guard), expectedGuard) << "prefix canary";
        EXPECT_EQ(std::vector<float>(output.begin() + guard + count, output.end()), expectedGuard) << "tail canary";
        const std::vector<float> actual(output.begin() + guard, output.begin() + guard + count);
        Verify(actual, expected, plan.c.type);
        if (repeat == 0) first = actual;
        else EXPECT_EQ(std::memcmp(first.data(), actual.data(), count * sizeof(float)), 0);
        if (!plan.c.inPlace) EXPECT_EQ(guardedX.Get<float>(count + 2 * guard), inputWithGuards);
        else EXPECT_EQ(guardedY.Get<float>(count + 2 * guard), blankOutput);
    }
    EXPECT_EQ(plan.dPtr.Get<int32_t>(plan.a.ptr.size()), plan.a.ptr);
    EXPECT_EQ(plan.dIdx.Get<int32_t>(plan.a.idx.size()), plan.a.idx);
    EXPECT_EQ(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    EXPECT_EQ(plan.dAlpha.Get<float>(plan.alpha.size()), plan.alpha);
}

class SpSVArch22Diagonal : public SpSVArch22Test, public ::testing::WithParamInterface<DiagonalCase> {};
TEST_P(SpSVArch22Diagonal, TailOwnersAndMulticoreAcrossUpdates)
{
    const auto test = GetParam();
    const auto &c = test.config;
    Plan plan(MakeDuplicateDiagonal(test), c);
    GuardedVectors vectors(plan);
    ASSERT_EQ(vectors.Bind(plan), kSuccess);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto x = Input(plan.a, c);
    const auto original = StoredValues(plan.a, c, false);
    {
        SCOPED_TRACE("initial solve");
        VerifyGuardedDiagonalTwice(plan, vectors,
            DiagonalGolden(plan.a, c, original, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    const auto general = StoredValues(plan.a, c, true);
    UpdateInput dGeneral(general, c.type);
    ASSERT_EQ(dGeneral.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    {
        SCOPED_TRACE("GENERAL update solve");
        VerifyGuardedDiagonalTwice(plan, vectors,
            DiagonalGolden(plan.a, c, general, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<Complex> diagonal;
    for (int row = 0; row < plan.a.m; ++row) diagonal.push_back(Value(10.0 + (row % 7) * 0.0625, -0.1875, c));
    UpdateInput dDiagonal(diagonal, c.type);
    ASSERT_EQ(dDiagonal.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    {
        SCOPED_TRACE("DIAGONAL update solve");
        VerifyGuardedDiagonalTwice(plan, vectors,
            DiagonalGolden(plan.a, c, general, x, plan.AlphaValue(), &diagonal));
    }
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    dGeneral.ExpectUnchanged();
    dDiagonal.ExpectUnchanged();
}

std::vector<DiagonalCase> DiagonalCases()
{
    std::vector<DiagonalCase> tests;
    const int sizes[] = {129, 129, 257, 257, 4097, 4097};
    const aclsparseOperation_t operations[] = {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
        ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
        ACL_SPARSE_OP_TRANSPOSE};
    for (int i = 0; i < 6; ++i) {
        Case c;
        c.type = i % 2 == 0 ? ACL_FLOAT : ACL_COMPLEX64;
        c.operation = operations[i];
        c.base = 1;
        c.lower = i % 2 == 0;
        c.deviceAlpha = i == 1 || i == 2 || i == 5;
        c.inPlace = i == 1 || i == 2 || i == 4;
        c.name = std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + "M" + std::to_string(sizes[i]) +
            "_" + "NTH"[static_cast<int>(c.operation)] + "_Base1" +
            (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") + (c.inPlace ? "_InPlace" : "_Separate");
        tests.push_back({c, sizes[i]});
    }
    return tests;
}
INSTANTIATE_TEST_SUITE_P(DiagonalOwners, SpSVArch22Diagonal, ::testing::ValuesIn(DiagonalCases()),
    [](const ::testing::TestParamInfo<DiagonalCase> &info) { return info.param.config.name; });

struct ChainCase {
    Case config;
    int rows;
};

Matrix MakeBandedChain(const ChainCase &test)
{
    Matrix a;
    a.m = test.rows;
    const auto &c = test.config;
    a.ptr.reserve(static_cast<size_t>(a.m) + 1);
    a.idx.reserve(static_cast<size_t>(a.m) * 4);
    a.entries.reserve(static_cast<size_t>(a.m) * 4);
    a.ptr.push_back(c.base);
    auto append = [&](int row, int col, Complex value) {
        a.idx.push_back(col + c.base);
        a.entries.push_back({row, col, value});
    };
    for (int row = 0; row < a.m; ++row) {
        // Dyadic, bounded coefficients keep the chain well conditioned in both
        // UNIT and NON_UNIT modes. Diagonals and dependencies are interleaved
        // deliberately, rather than supplying sorted columns to analysis.
        append(row, row, Value(3.0 + (row % 8) * 0.03125, 0.125, c));
        const int second = c.lower ? row - 2 : row + 2;
        if (second >= 0 && second < a.m) append(row, second, Value(-0.03125, 0.0078125, c));
        append(row, row, Value(2.0, -0.0625, c));
        const int first = c.lower ? row - 1 : row + 1;
        if (first >= 0 && first < a.m) append(row, first,
            Value(row % 2 == 0 ? -0.0625 : 0.0625, row % 3 == 0 ? -0.015625 : 0.015625, c));
        a.ptr.push_back(c.base + static_cast<int32_t>(a.entries.size()));
    }
    return a;
}

struct SparseReferenceMatrix {
    std::vector<int32_t> rowPtr, columns;
    std::vector<Complex> values;
};
SparseReferenceMatrix NormalizeReference(const Matrix &a, const Case &c, const std::vector<Complex> &values)
{
    // Normalize op(A) by counting/scattering into row groups, preserving the
    // source storage order inside every effective row. All arrays are O(m+nnz).
    // This reference uses FP64 / complex128 throughout, independently of the
    // DD threshold and the device Y-cache capacity.
    std::vector<int32_t> rowPtr(static_cast<size_t>(a.m) + 1, 0);
    auto selected = [&](const Entry &e) {
        return e.row >= 0 && (c.lower ? e.col <= e.row : e.col >= e.row);
    };
    const bool transpose = c.operation != ACL_SPARSE_OP_NON_TRANSPOSE;
    for (const auto &e : a.entries) {
        if (selected(e)) ++rowPtr[(transpose ? e.col : e.row) + 1];
    }
    for (int row = 0; row < a.m; ++row) rowPtr[row + 1] += rowPtr[row];
    std::vector<int32_t> columns(rowPtr.back());
    std::vector<Complex> effectiveValues(rowPtr.back());
    auto next = rowPtr;
    for (size_t p = 0; p < a.entries.size(); ++p) {
        const auto &e = a.entries[p];
        if (!selected(e)) continue;
        const int row = transpose ? e.col : e.row;
        const int destination = next[row]++;
        columns[destination] = transpose ? e.row : e.col;
        effectiveValues[destination] = c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE ?
            std::conj(values[p]) : values[p];
    }
    return {std::move(rowPtr), std::move(columns), std::move(effectiveValues)};
}

std::vector<Complex> SparseChainGolden(const Matrix &a, const Case &c,
    const std::vector<Complex> &values, const std::vector<Complex> &x, Complex alpha,
    const std::vector<Complex> *replacement = nullptr)
{
    const auto effective = NormalizeReference(a, c, values);
    const auto &rowPtr = effective.rowPtr;
    const auto &columns = effective.columns;
    const auto &effectiveValues = effective.values;
    const bool forward = c.operation != ACL_SPARSE_OP_NON_TRANSPOSE ? !c.lower : c.lower;
    std::vector<Complex> y(a.m);
    for (int step = 0; step < a.m; ++step) {
        const int row = forward ? step : a.m - 1 - step;
        Complex sum{};
        Complex diagonal{};
        for (int32_t p = rowPtr[row]; p < rowPtr[row + 1]; ++p) {
            if (columns[p] == row) diagonal += effectiveValues[p];
            else {
                // Visit every stored dependency, including explicit zero.
                // Thus 0 * Inf retains IEEE NaN propagation; absent entries
                // introduce neither arithmetic nor a dense reference scan.
                sum += effectiveValues[p] * y[columns[p]];
            }
        }
        if (replacement != nullptr) {
            diagonal = (*replacement)[row];
            if (c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) diagonal = std::conj(diagonal);
        }
        if (c.unit) diagonal = Complex{1.0, 0.0};
        // Component-wise products preserve the specified IEEE propagation.
        // Some std::complex implementations recover infinities differently.
        const Complex scaled = c.type == ACL_FLOAT ? Complex{alpha.real() * x[row].real(), 0.0} :
            Complex{alpha.real() * x[row].real() - alpha.imag() * x[row].imag(),
                alpha.real() * x[row].imag() + alpha.imag() * x[row].real()};
        const auto rhs = scaled - sum;
        y[row] = ReferenceDivide(rhs, diagonal, c.type);
    }
    return y;
}

class SpSVArch22Banded : public SpSVArch22Test, public ::testing::WithParamInterface<ChainCase> {};
TEST_P(SpSVArch22Banded, DdBoundaryAndYCacheWindowAcrossUpdates)
{
    const auto test = GetParam();
    const auto &c = test.config;
    Plan plan(MakeBandedChain(test), c);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto x = Input(plan.a, c);
    const auto original = StoredValues(plan.a, c, false);
    {
        SCOPED_TRACE("initial solve");
        plan.VerifyTwice(SparseChainGolden(plan.a, c, original, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    const auto general = StoredValues(plan.a, c, true);
    UpdateInput dGeneral(general, c.type);
    ASSERT_EQ(dGeneral.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    {
        SCOPED_TRACE("GENERAL update solve");
        plan.VerifyTwice(SparseChainGolden(plan.a, c, general, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<Complex> diagonal;
    diagonal.reserve(plan.a.m);
    for (int row = 0; row < plan.a.m; ++row) diagonal.push_back(Value(8.0 + (row % 16) * 0.0625, -0.125, c));
    UpdateInput dDiagonal(diagonal, c.type);
    ASSERT_EQ(dDiagonal.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    {
        SCOPED_TRACE("DIAGONAL update solve");
        plan.VerifyTwice(SparseChainGolden(plan.a, c, general, x, plan.AlphaValue(), &diagonal));
    }
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    dGeneral.ExpectUnchanged();
    dDiagonal.ExpectUnchanged();
}

std::vector<ChainCase> ChainCases()
{
    std::vector<ChainCase> tests;
    const int sizes[] = {512, 513, 256, 257, 24577, 12289, 6400};
    const aclDataType types[] = {ACL_FLOAT, ACL_FLOAT, ACL_COMPLEX64, ACL_COMPLEX64,
        ACL_FLOAT, ACL_COMPLEX64, ACL_COMPLEX64};
    const aclsparseOperation_t operations[] = {ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_TRANSPOSE,
        ACL_SPARSE_OP_CONJUGATE_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE};
    for (int i = 0; i < 7; ++i) {
        Case c;
        c.type = types[i];
        c.operation = operations[i];
        c.lower = i == 0 || i == 2 || i == 4;
        c.unit = i == 6;
        c.base = i == 0 || i == 3 || i == 5 ? 0 : 1;
        c.deviceAlpha = i == 1 || i == 2 || i == 5;
        c.inPlace = i == 1 || i == 3 || i == 5;
        c.name = std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + "M" + std::to_string(sizes[i]) +
            "_" + "NTH"[static_cast<int>(c.operation)] + (c.lower ? "_Lower" : "_Upper") +
            (c.unit ? "_Unit" : "_Nonunit") + "_Base" + std::to_string(c.base) +
            (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") + (c.inPlace ? "_InPlace" : "_Separate");
        tests.push_back({c, sizes[i]});
    }
    return tests;
}
INSTANTIATE_TEST_SUITE_P(GeneralChains, SpSVArch22Banded, ::testing::ValuesIn(ChainCases()),
    [](const ::testing::TestParamInfo<ChainCase> &info) { return info.param.config.name; });

struct ResidentCase {
    Case config;
    int rows;
    bool longRow;
};

Matrix MakeLongDependencyRow(const ResidentCase &test)
{
    Matrix a;
    a.m = test.rows;
    const auto &c = test.config;
    const int target = c.lower ? a.m - 1 : 0;
    a.ptr.push_back(c.base);
    auto append = [&](int row, int col, Complex value) {
        a.idx.push_back(col + c.base);
        a.entries.push_back({row, col, value});
    };
    for (int row = 0; row < a.m; ++row) {
        if (row == target) {
            for (int dependency = 0; dependency < a.m - 1; ++dependency) {
                // Separate the duplicate diagonal entries with thousands of
                // dependencies, so span changes cannot reset diagonal state.
                if (dependency == 17) append(row, row, Value(3.0, 0.125, c));
                if (dependency == (a.m - 1) / 2) append(row, row, Value(2.0, -0.0625, c));
                const int col = c.lower ? a.m - 2 - dependency : a.m - 1 - dependency;
                append(row, col, Value(dependency % 2 == 0 ? 0.00006103515625 : -0.00006103515625,
                    dependency % 3 == 0 ? -0.0000152587890625 : 0.0000152587890625, c));
            }
        } else {
            append(row, row, Value(3.0 + (row % 8) * 0.03125, 0.125, c));
            append(row, row, Value(2.0, -0.0625, c));
        }
        a.ptr.push_back(c.base + static_cast<int32_t>(a.entries.size()));
    }
    return a;
}

void VerifyResidentBoundary(const ResidentCase &test, const Matrix &matrix)
{
    const auto &c = test.config;
    const int components = c.type == ACL_COMPLEX64 ? 2 : 1;
    if (!test.longRow) ASSERT_EQ(matrix.m * components, 24576);
    else {
        const int row = c.lower ? matrix.m - 1 : 0;
        const int32_t begin = matrix.ptr[row] - c.base;
        const int32_t end = matrix.ptr[row + 1] - c.base;
        ASSERT_GT(end - begin, 4096);
        ASSERT_GT((end - 1) / 4096, begin / 4096);
        ASSERT_GT(((end - 1) * components) / 8192, (begin * components) / 8192);
        ASSERT_NE(end % 4096, 0);
        ASSERT_NE((end * components) % 8192, 0);
    }
}
class SpSVArch22Resident : public SpSVArch22Test, public ::testing::WithParamInterface<ResidentCase> {};
TEST_P(SpSVArch22Resident, FullYCapacityAndLongRowReadSpansAcrossUpdates)
{
    const auto test = GetParam();
    const auto &c = test.config;
    const auto matrix = test.longRow ? MakeLongDependencyRow(test) : MakeBandedChain({c, test.rows});
    VerifyResidentBoundary(test, matrix);
    if (::testing::Test::HasFatalFailure()) return;
    Plan plan(matrix, c);
    GuardedVectors vectors(plan);
    ASSERT_EQ(vectors.Bind(plan), kSuccess);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto x = Input(plan.a, c);
    const auto original = StoredValues(plan.a, c, false);
    {
        SCOPED_TRACE("initial solve");
        // The existing guarded verifier is matrix-independent: it checks
        // complete output accuracy, bitwise repeats, input data and canaries.
        VerifyGuardedDiagonalTwice(plan, vectors,
            SparseChainGolden(plan.a, c, original, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    const auto general = StoredValues(plan.a, c, true);
    UpdateInput dGeneral(general, c.type);
    ASSERT_EQ(dGeneral.Apply(plan, ACL_SPARSE_SPSV_UPDATE_GENERAL), kSuccess);
    {
        SCOPED_TRACE("GENERAL update solve");
        VerifyGuardedDiagonalTwice(plan, vectors,
            SparseChainGolden(plan.a, c, general, x, plan.AlphaValue()));
    }
    if (::testing::Test::HasFatalFailure()) return;

    std::vector<Complex> diagonal;
    diagonal.reserve(plan.a.m);
    for (int row = 0; row < plan.a.m; ++row) diagonal.push_back(Value(8.0 + (row % 16) * 0.0625, -0.125, c));
    UpdateInput dDiagonal(diagonal, c.type);
    ASSERT_EQ(dDiagonal.Apply(plan, ACL_SPARSE_SPSV_UPDATE_DIAGONAL), kSuccess);
    {
        SCOPED_TRACE("DIAGONAL update solve");
        VerifyGuardedDiagonalTwice(plan, vectors,
            SparseChainGolden(plan.a, c, general, x, plan.AlphaValue(), &diagonal));
    }
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    dGeneral.ExpectUnchanged();
    dDiagonal.ExpectUnchanged();
}

std::vector<ResidentCase> ResidentCases()
{
    std::vector<ResidentCase> tests;
    const int sizes[] = {24576, 12288, 8301, 4353};
    const aclsparseOperation_t operations[] = {ACL_SPARSE_OP_TRANSPOSE, ACL_SPARSE_OP_CONJUGATE_TRANSPOSE,
        ACL_SPARSE_OP_NON_TRANSPOSE, ACL_SPARSE_OP_NON_TRANSPOSE};
    for (int i = 0; i < 4; ++i) {
        Case c;
        c.type = i % 2 == 0 ? ACL_FLOAT : ACL_COMPLEX64;
        c.lower = i % 2 == 0;
        c.operation = operations[i];
        c.base = i == 0 || i == 3 ? 0 : 1;
        c.deviceAlpha = i == 1 || i == 2;
        c.inPlace = i == 1 || i == 2;
        // Long F32 row spans [16600,24902), crossing both col/4096 and
        // value/8192 windows with a tail. Long C64 row spans [0,4354),
        // crossing col/4096 and value-float/8192 together with a tail.
        const bool longRow = i >= 2;
        c.name = std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + "M" + std::to_string(sizes[i]) +
            (longRow ? "_LongRow" : "_FullY") + "_" + "NTH"[static_cast<int>(c.operation)] +
            (c.lower ? "_Lower" : "_Upper") + "_Base" + std::to_string(c.base) +
            (c.deviceAlpha ? "_DeviceAlpha" : "_HostAlpha") + (c.inPlace ? "_InPlace" : "_Separate");
        tests.push_back({c, sizes[i], longRow});
    }
    return tests;
}
INSTANTIATE_TEST_SUITE_P(ResidentSpans, SpSVArch22Resident, ::testing::ValuesIn(ResidentCases()),
    [](const ::testing::TestParamInfo<ResidentCase> &info) { return info.param.config.name; });

const char *FormatName(aclsparseFormat_t format)
{
    return format == ACL_SPARSE_FORMAT_CSR ? "CSR" : format == ACL_SPARSE_FORMAT_CSC ? "CSC" :
        format == ACL_SPARSE_FORMAT_COO ? "COO" : "SELL";
}
std::string CombinationKey(const Case &c, int update)
{
    return std::string(c.type == ACL_FLOAT ? "F32_" : "C64_") + FormatName(c.format) + "_" +
        "NTH"[static_cast<int>(c.operation)] + (c.lower ? "_Lower" : "_Upper") +
        (c.unit ? "_Unit" : "_Nonunit") + "_Base" + std::to_string(c.base) +
        (c.deviceAlpha ? "_Device" : "_Host") + (c.inPlace ? "_Inplace" : "_Separate") +
        (update == 0 ? "_GENERAL" : "_DIAGONAL");
}

Matrix EncodeEntries(int m, const std::vector<Entry> &source, const Case &c)
{
    Matrix a;
    a.m = m;
    if (c.format == ACL_SPARSE_FORMAT_COO) {
        for (auto it = source.rbegin(); it != source.rend(); ++it) {
            a.ptr.push_back(it->row + c.base);
            a.idx.push_back(it->col + c.base);
            a.entries.push_back(*it);
        }
        return a;
    }
    std::vector<std::vector<Entry>> groups(m);
    for (const auto &e : source) groups[c.format == ACL_SPARSE_FORMAT_CSC ? e.col : e.row].push_back(e);
    for (auto &group : groups) std::reverse(group.begin(), group.end());
    if (c.format == ACL_SPARSE_FORMAT_SLICED_ELL) {
        for (const auto &group : groups) a.width = std::max(a.width, static_cast<int>(group.size()));
        a.ptr = {c.base, c.base + m * a.width};
        for (const auto &group : groups) {
            for (const auto &e : group) { a.idx.push_back(e.col + c.base); a.entries.push_back(e); }
            for (size_t slot = group.size(); slot < static_cast<size_t>(a.width); ++slot) {
                a.idx.push_back(-1);
                a.entries.push_back({-1, -1, Value(999.0, -999.0, c)});
            }
        }
    } else {
        a.ptr.push_back(c.base);
        for (const auto &group : groups) {
            for (const auto &e : group) {
                a.idx.push_back((c.format == ACL_SPARSE_FORMAT_CSC ? e.row : e.col) + c.base);
                a.entries.push_back(e);
            }
            a.ptr.push_back(c.base + static_cast<int32_t>(a.entries.size()));
        }
    }
    return a;
}

struct SampleCounts {
    // Channels: A real, A imaginary, X real, X imaginary.
    // Buckets: raw uniform, raw normal, constructed special.
    std::array<std::array<size_t, 3>, 4> count{};
};
struct CombinationData {
    Matrix a;
    std::vector<Complex> x;
    SampleCounts sampling;
    int missingRow = -1;
};
void InitializeRandomData(CombinationData &data, const Case &c, int regime, uint32_t seed)
{
    data.a = MakeMatrix(c);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int bucket = regime < 7 ? 0 : 1;
    auto draw = [&]() {
        return bucket == 0 ? uniform(generator) : normal(generator);
    };
    for (auto &e : data.a.entries) {
        if (e.row < 0) continue; // Skip SELL padding.
        float real = draw(), imag = c.type == ACL_COMPLEX64 ? draw() : 0.0f;
        e.value = Complex{real, imag};
    }
    for (int row = 0; row < data.a.m; ++row) {
        const float real = draw();
        const float imag = c.type == ACL_COMPLEX64 ? draw() : 0.0f;
        data.x.emplace_back(real, imag);
    }
}
void InitializeSpecialData(CombinationData &data, const Case &c, size_t combination)
{
    const int kind = static_cast<int>(combination % 5);
    constexpr int m = 5;
    std::vector<Entry> source;
    for (int row = 0; row < m; ++row) {
        if (kind == 2 && row == 2) {
            data.missingRow = row;
            continue;
        }
        const double diagonal = kind == 1 && row == 2 ? 0.0 : 0.5;
        source.push_back({row, row, Value(diagonal, 0.0, c)});
        source.push_back({row, row, Value(diagonal, 0.0, c)});
    }
    data.a = EncodeEntries(m, source, c);
    for (int row = 0; row < m; ++row) {
        // Zero complex RHS gives unambiguous NaN/NaN for zero division.
        auto x = Value(kind == 1 || kind == 2 ? 0.0 : 0.5 + row * 0.25, 0.0, c);
        if (row == m - 1 && kind == 3) {
            x = Value(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), c);
        }
        if (row == m - 1 && kind == 4) x = Value(std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(), c);
        data.x.push_back(x);
    }
}
CombinationData MakeCombinationData(const Case &c, int regime, uint32_t seed, size_t combination)
{
    CombinationData data;
    if (regime < 9) InitializeRandomData(data, c, regime, seed);
    else InitializeSpecialData(data, c, combination);
    return data;
}

std::vector<Complex> CombinationUpdateValues(const Matrix &a, const Case &c, int regime, int update,
    uint32_t seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int bucket = regime < 7 ? 0 : regime < 9 ? 1 : 2;
    auto draw = [&]() {
        return bucket == 0 ? uniform(generator) : normal(generator);
    };
    const size_t count = update == 0 ? a.entries.size() : static_cast<size_t>(a.m);
    std::vector<Complex> out;
    out.reserve(count);
    for (size_t p = 0; p < count; ++p) {
        if (update == 0 && a.entries[p].row < 0) {
            out.push_back(a.entries[p].value);
            continue;
        }
        if (bucket != 2) {
            const float real = draw();
            const float imag = c.type == ACL_COMPLEX64 ? draw() : 0.0f;
            out.emplace_back(real, imag);
        } else {
            out.push_back(update == 0 ? -a.entries[p].value : Value(1.0, -0.125, c));
        }
    }
    return out;
}
void ExpectBitwiseFloats(const std::vector<float> &actual, const std::vector<float> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    if (!actual.empty()) EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
}
void VerifyRepeatedSolve(Plan &plan, const std::vector<Complex> &expected)
{
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.dX.Put(plan.input);
        const auto solve = plan.Solve(plan.c.operation);
        const auto sync = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        ASSERT_EQ(solve, kSuccess);
        ASSERT_EQ(sync, ACL_SUCCESS);
        auto actual = (plan.c.inPlace ? plan.dX : plan.dY).Get<float>(plan.input.size());
        Verify(actual, expected, plan.c.type); // Same existing tolerances and IEEE rules.
        if (repeat == 0)
            first = actual;
        else
            ExpectBitwiseFloats(actual, first);
        if (!plan.c.inPlace) ExpectBitwiseFloats(plan.dX.Get<float>(plan.input.size()), plan.input);
    }
    EXPECT_EQ(plan.dPtr.Get<int32_t>(plan.a.ptr.size()), plan.a.ptr);
    EXPECT_EQ(plan.dIdx.Get<int32_t>(plan.a.idx.size()), plan.a.idx);
    ExpectBitwiseFloats(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    ExpectBitwiseFloats(plan.dAlpha.Get<float>(plan.alpha.size()), plan.alpha);
}
void DestroyChecked(Plan &plan)
{
    CheckSparse(aclsparseSpSV_destroyDescr(plan.state), "destroy plan");
    plan.state = nullptr;
    CheckSparse(aclsparseDestroyDnVec(plan.x), "destroy X");
    plan.x = nullptr;
    CheckSparse(aclsparseDestroyDnVec(plan.y), "destroy Y");
    plan.y = nullptr;
    CheckSparse(aclsparseDestroySpMat(plan.mat), "destroy matrix");
    plan.mat = nullptr;
}

struct CombinationScenario {
    const Case &config;
    int regime, update;
    size_t combination;
    uint32_t seed;
    bool bound;
};
void ExecuteCombinationPlan(const CombinationScenario &test, const CombinationData &data)
{
    const auto &c = test.config;
    Plan plan(data.a, c);
    plan.input = Pack(data.x, c.type);
    plan.dX.Put(plan.input);
    size_t workspaceBytes = 0;
    CheckSparse(plan.Size(&workspaceBytes, nullptr, nullptr, c.operation, c.type), "NULL-vector bufferSize");
    CheckSparse(plan.Analyze(test.bound), "analysis");
    const auto original = StoredValues(data.a, c, false);
    VerifyRepeatedSolve(plan, SparseChainGolden(data.a, c, original, data.x, plan.AlphaValue()));
    if (!::testing::Test::HasFatalFailure()) {
        auto replacement = CombinationUpdateValues(data.a, c, test.regime, test.update,
            test.seed ^ 0x5a17c93du);
        const auto packed = Pack(replacement, c.type);
        DeviceBuffer dReplacement(packed.size() * sizeof(float));
        dReplacement.Put(packed);
        CheckSparse(aclsparseSpSV_updateMatrix(
                        SpSVArch22Test::handle, plan.state, dReplacement.get(),
                        test.update == 0 ? ACL_SPARSE_SPSV_UPDATE_GENERAL : ACL_SPARSE_SPSV_UPDATE_DIAGONAL),
                    "updateMatrix");
        if (test.update == 1 && data.missingRow >= 0) replacement[data.missingRow] = Complex{};
        const auto expected = test.update == 0 ? SparseChainGolden(data.a, c, replacement, data.x, plan.AlphaValue()) :
            SparseChainGolden(data.a, c, original, data.x, plan.AlphaValue(), &replacement);
        VerifyRepeatedSolve(plan, expected);
        CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "finish update/solve before release");
        ExpectBitwiseFloats(dReplacement.Get<float>(packed.size()), packed);
    }
    CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "finish before destroy");
    DestroyChecked(plan);
}
void RunCombinationRegime(const CombinationScenario &test)
{
    const auto data = MakeCombinationData(test.config, test.regime, test.seed, test.combination);
    SCOPED_TRACE(test.config.name + " regime=" + std::to_string(test.regime) + " seed=" + std::to_string(test.seed));
    try {
        ExecuteCombinationPlan(test, data);
    } catch (const std::exception &exception) {
        ADD_FAILURE() << exception.what();
    }
}
std::vector<std::pair<Case, int>> AttributeCombinations()
{
    std::vector<std::pair<Case, int>> cases;
    for (const auto &core : FunctionalCases()) for (int pointer = 0; pointer < 2; ++pointer)
        for (int alias = 0; alias < 2; ++alias) for (int update = 0; update < 2; ++update) {
                Case c = core;
                c.deviceAlpha = pointer != 0;
                c.inPlace = alias != 0;
                c.name = CombinationKey(c, update);
                cases.emplace_back(c, update);
        }
    return cases;
}
class SpSVArch22Matrix : public SpSVArch22Test {};
TEST_F(SpSVArch22Matrix, All1536LegalCombinationsWithSevenTwoOneRegimes)
{
    std::set<std::string> combinations;
    size_t combination = 0;
    std::array<size_t, 3> regimes{};
    for (const auto &test : AttributeCombinations()) {
        ASSERT_TRUE(combinations.insert(test.first.name).second);
        for (int regime = 0; regime < 10; ++regime) {
            const uint32_t seed = 0x53505600u + static_cast<uint32_t>(combination * 10 + regime);
            RunCombinationRegime({test.first, regime, test.second, combination, seed, regime % 2 != 0});
            ++regimes[regime < 7 ? 0 : regime < 9 ? 1 : 2];
            if (::testing::Test::HasFatalFailure()) return;
        }
        ++combination;
    }
    EXPECT_EQ(combinations.size(), 1536u);
    EXPECT_EQ(combination, 1536u);
    EXPECT_EQ(regimes[0], 10752u);
    EXPECT_EQ(regimes[1], 3072u);
    EXPECT_EQ(regimes[2], 1536u);
}

struct WorkspaceLayout {
    size_t bytes = 512, row = 0, col = 0, value = 0, perm = 0, solution = 0;
};
WorkspaceLayout ExpectedWorkspaceLayout(const Matrix& a, aclDataType type, bool unit)
{
    WorkspaceLayout layout;
    auto append = [&](size_t count, size_t elementBytes, size_t &offset) {
        offset = layout.bytes;
        layout.bytes += (count * elementBytes + 63) / 64 * 64;
    };
    append(static_cast<size_t>(a.m) + 1, sizeof(int32_t), layout.row);
    append(a.entries.size(), sizeof(int32_t), layout.col);
    append(a.entries.size(), type == ACL_COMPLEX64 ? 2 * sizeof(float) : sizeof(float), layout.value);
    append(a.entries.size(), sizeof(int32_t), layout.perm);
    size_t scratch = 0;
    append(a.m, sizeof(int32_t), scratch);
    const size_t components = type == ACL_COMPLEX64 ? 2 : 1;
    const auto terms = a.m * components <= spsv_a2_ub::kResidentCapacity ? spsv_a2_ub::kWideCompactTerms :
                                                                           spsv_a2_ub::kWideRegularTerms;
    append(a.m, components * spsv_a2_ub::SolutionWords(terms) * sizeof(float), layout.solution);
    size_t input = 0;
    append(a.m, (type == ACL_COMPLEX64 ? 2 : 1) * sizeof(float), input);
    if (unit && a.m * (type == ACL_COMPLEX64 ? 2 : 1) <= spsv_a2_ub::kResidentCapacity)
        append(a.m, (type == ACL_COMPLEX64 ? 2 : 1) * spsv_a2_ub::kUnitPrecisionStride * sizeof(float), input);
    return layout;
}
template<class T> std::vector<T> ReadDeviceAt(const void *base, size_t byteOffset, size_t count)
{
    std::vector<T> out(count);
    if (count != 0)
        CheckAcl(aclrtMemcpy(out.data(), count * sizeof(T), static_cast<const uint8_t*>(base) + byteOffset,
                             count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST),
                 "read workspace");
    return out;
}
struct NormalizedExpected {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    std::vector<int32_t> permutation;
    std::vector<float> values;
    bool diagonal = true;
};
NormalizedExpected ExpectedNormalization(const Matrix &a, const Case &c)
{
    NormalizedExpected expected;
    std::vector<std::vector<std::pair<int32_t, size_t>>> rows(a.m);
    for (size_t p = 0; p < a.entries.size(); ++p) {
        const auto &e = a.entries[p];
        if (e.row < 0 || (c.lower ? e.col > e.row : e.col < e.row)) continue;
        if (e.row != e.col) expected.diagonal = false;
        const bool transpose = c.operation != ACL_SPARSE_OP_NON_TRANSPOSE;
        rows[transpose ? e.col : e.row].push_back({transpose ? e.row : e.col, p});
    }
    expected.rows.push_back(0);
    for (const auto &row : rows) {
        for (const auto &entry : row) {
            expected.columns.push_back(entry.first);
            expected.permutation.push_back(static_cast<int32_t>(entry.second));
            auto value = a.entries[entry.second].value;
            if (c.type == ACL_COMPLEX64 && c.operation == ACL_SPARSE_OP_CONJUGATE_TRANSPOSE) value = std::conj(value);
            auto packed = Pack({value}, c.type);
            expected.values.insert(expected.values.end(), packed.begin(), packed.end());
        }
        expected.rows.push_back(static_cast<int32_t>(expected.columns.size()));
    }
    return expected;
}
Matrix MakeNormalizationMatrix(const Case &c)
{
    constexpr int m = 17;
    const int empty = 8;
    const int longRow = c.lower ? m - 1 : 0;
    std::vector<Entry> entries;
    for (int row = 0; row < m; ++row) {
        if (row == empty) continue;
        entries.push_back({row, row, Value(3.0 + row * 0.0625, 0.25, c)});
        if (row == longRow) for (int col = m - 1; col >= 0; --col) {
            if (col != row && col != empty) entries.push_back({row, col, Value((col % 2 ? -1.0 : 1.0) * 0.0625,
                (col % 3 ? 1.0 : -1.0) * 0.03125, c)});
        }
        entries.push_back({row, row, Value(2.0, -0.125, c)});
    }
    // Opposite-triangle data forces filtering, and must not appear in row/perm.
    const int ignoredRow = c.lower ? 0 : m - 1;
    entries.push_back({ignoredRow, c.lower ? m - 1 : 0, Value(93.0, -7.0, c)});
    return EncodeEntries(m, entries, c);
}

class SpSVArch22Normalization : public SpSVArch22Test {};
void VerifyNormalizationGuards(const DeviceBuffer& allocation, const WorkspaceLayout& layout, size_t guard,
                               const std::vector<uint8_t>& blank)
{
    const auto observed = allocation.Get<uint8_t>(blank.size());
    EXPECT_EQ(std::vector<uint8_t>(observed.begin() + guard + layout.solution, observed.begin() + guard + layout.bytes),
              std::vector<uint8_t>(layout.bytes - layout.solution, 0xa5));
    EXPECT_EQ(std::vector<uint8_t>(observed.begin(), observed.begin() + guard), std::vector<uint8_t>(guard, 0xa5));
    EXPECT_EQ(std::vector<uint8_t>(observed.begin() + guard + layout.bytes, observed.end()),
              std::vector<uint8_t>(guard, 0xa5));
}
void InspectNormalization(const Case &c, const Matrix &a, const NormalizedExpected &expected,
    const WorkspaceLayout &layout)
{
    Plan plan(a, c);
    size_t queried = 0;
    CheckSparse(plan.Size(&queried, nullptr, nullptr, c.operation, c.type), "normalization bufferSize");
    if (queried != layout.bytes) throw std::runtime_error("Workspace schema differs from inspected arch22 layout");
    constexpr size_t guard = 512;
    DeviceBuffer allocation(layout.bytes + 2 * guard);
    const std::vector<uint8_t> blank(layout.bytes + 2 * guard, 0xa5);
    void *workspace = static_cast<uint8_t *>(allocation.get()) + guard;
    std::vector<int32_t> previousRows;
    std::vector<int32_t> previousCols;
    std::vector<int32_t> previousPerm;
    std::vector<float> previousValues;
    for (int repeat = 0; repeat < 2; ++repeat) {
        allocation.Put(blank);
        CheckSparse(aclsparseSpSV_analysis(SpSVArch22Test::handle, c.operation, plan.Alpha(), plan.mat, nullptr,
                                           nullptr, c.type, kDefault, plan.state, workspace),
                    "guarded analysis");
        CheckAcl(aclrtSynchronizeStream(SpSVArch22Test::env->stream()), "normalization complete");
        const auto status = ReadDeviceAt<int32_t>(workspace, 0, 3);
        EXPECT_EQ(status[0], 0);
        EXPECT_EQ(status[1], static_cast<int32_t>(expected.columns.size()));
        EXPECT_EQ(status[2], expected.diagonal ? 1 : 0);
        const auto rows = ReadDeviceAt<int32_t>(workspace, layout.row, expected.rows.size());
        const auto columns = ReadDeviceAt<int32_t>(workspace, layout.col, expected.columns.size());
        const auto values = ReadDeviceAt<float>(workspace, layout.value, expected.values.size());
        const auto permutation = ReadDeviceAt<int32_t>(workspace, layout.perm, expected.permutation.size());
        EXPECT_EQ(rows, expected.rows);
        EXPECT_EQ(columns, expected.columns);
        EXPECT_EQ(permutation, expected.permutation);
        ExpectBitwiseFloats(values, expected.values);
        if (repeat != 0) {
            EXPECT_EQ(rows, previousRows);
            EXPECT_EQ(columns, previousCols);
            EXPECT_EQ(permutation, previousPerm);
            ExpectBitwiseFloats(values, previousValues);
        }
        previousRows = rows;
        previousCols = columns;
        previousValues = values;
        previousPerm = permutation;
        VerifyNormalizationGuards(allocation, layout, guard, blank);
        EXPECT_EQ(plan.dPtr.Get<int32_t>(a.ptr.size()), a.ptr);
        EXPECT_EQ(plan.dIdx.Get<int32_t>(a.idx.size()), a.idx);
        ExpectBitwiseFloats(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
    }
    // Descriptor destruction also precedes release of its external allocation.
    DestroyChecked(plan);
}
void RunNormalizationCase(Case c)
{
    c.name = CombinationKey(c, 0);
    const auto a = MakeNormalizationMatrix(c);
    const auto expected = ExpectedNormalization(a, c);
    const auto layout = ExpectedWorkspaceLayout(a, c.type, c.unit);
    SCOPED_TRACE(c.name);
    try {
        InspectNormalization(c, a, expected, layout);
    } catch (const std::exception &exception) { ADD_FAILURE() << exception.what(); }
}
TEST_F(SpSVArch22Normalization, AllFormatsInspectAnalysisArraysAndGuardedWorkspace)
{
    constexpr aclsparseFormat_t formats[] = {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_CSC, ACL_SPARSE_FORMAT_COO,
                                             ACL_SPARSE_FORMAT_SLICED_ELL};
    size_t id = 0;
    for (; id < 96; ++id) {
        Case c;
        c.format = formats[id / 24];
        c.lower = (id / 12) % 2 != 0;
        c.operation = static_cast<aclsparseOperation_t>((id / 4) % 3);
        c.type = (id / 2) % 2 == 0 ? ACL_FLOAT : ACL_COMPLEX64;
        c.base = static_cast<int>(id % 2);
        RunNormalizationCase(c);
        if (::testing::Test::HasFatalFailure())
            return;
    }
    EXPECT_EQ(id, 96u);
}

class SpSVArch22ReleasedBufferChild : public SpSVArch22Test {};
TEST_F(SpSVArch22ReleasedBufferChild, DISABLED_RejectReleasedWorkspaceBeforeKernelLaunch)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(), kSuccess);
    ASSERT_EQ(aclrtSynchronizeStream(env->stream()), ACL_SUCCESS);
    int32_t currentDevice = -1;
    ASSERT_EQ(aclrtGetDevice(&currentDevice), ACL_SUCCESS);
    const void *releasedAddress = plan.workspace->get();
    ASSERT_EQ(spsv_memory::ValidateRange(releasedAddress, sizeof(float)), ACL_SUCCESS)
        << "Child requires a supported public allocation query before releasing workspace";
    ASSERT_EQ(plan.workspace->FreeNow(), ACL_SUCCESS);
    // No further device allocations occur before the binding checks. This
    // preflight mirrors Host DevicePointer, including its allocation query.
    // A retained virtual-address attribute alone is not allocation liveness.
    aclrtPtrAttributes attributes{};
    const auto query = aclrtPointerGetAttributes(releasedAddress, &attributes);
    const bool stillDevice = query == ACL_SUCCESS && attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
        attributes.location.id == static_cast<uint32_t>(currentDevice);
    void *allocation = nullptr;
    size_t allocationBytes = 0;
    const auto allocationQuery = aclrtMemGetAddressRange(
        const_cast<void *>(releasedAddress), &allocation, &allocationBytes);
    if (stillDevice && (allocationQuery == ACL_SUCCESS || allocationQuery == ACL_ERROR_RT_FEATURE_NOT_SUPPORT)) {
        ADD_FAILURE() << "Public queries cannot establish rejection of the released workspace; "
            "Solve/Update not called to avoid a use-after-free kernel. The requested rejection cannot be established.";
        return;
    }
    const auto solve = plan.Solve(c.operation);
    ASSERT_EQ(solve, kInvalid);
    const auto update = aclsparseSpSV_updateMatrix(
        handle, plan.state, plan.dValues.get(), ACL_SPARSE_SPSV_UPDATE_GENERAL);
    EXPECT_EQ(update, kInvalid);
}

// Fresh exec gives the negative lifetime probe its own ACL process/context.
// A fork-only death test would inherit the parent's runtime worker threads.
class SpSVArch22Lifecycle : public SpSVArch22Test {};
TEST_F(SpSVArch22Lifecycle, ReleasedWorkspaceRejectionInFreshProcess)
{
    DeviceBuffer probe(4096);
    const auto capability = spsv_memory::ValidateRange(probe.get(), 4096);
    if (capability == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        GTEST_SKIP() << "Public allocation range query is unsupported; released-workspace rejection is not guaranteed";
    }
    ASSERT_EQ(capability, ACL_SUCCESS);
    char executable[] = "/proc/self/exe";
    char filter[] = "--gtest_filter=SpSVArch22ReleasedBufferChild.DISABLED_RejectReleasedWorkspaceBeforeKernelLaunch";
    char enable[] = "--gtest_also_run_disabled_tests";
    char *arguments[] = {executable, filter, enable, nullptr};
    pid_t child = -1;
    const int spawned = posix_spawn(&child, executable, nullptr, nullptr, arguments, ::environ);
    ASSERT_EQ(spawned, 0);
    int result = 0;
    const auto waited = waitpid(child, &result, 0);
    EXPECT_EQ(waited, child);
    EXPECT_TRUE(WIFEXITED(result));
    const int exited = waited == child && WIFEXITED(result) ? WEXITSTATUS(result) : -1;
    EXPECT_EQ(exited, 0) << "Released-buffer probe failed; see isolated child output";
}

class SpSVArch22Lifetime : public SpSVArch22Test {};
TEST_F(SpSVArch22Lifetime, SolveCompletesBeforeExternalWorkspaceIsReleased)
{
    Case c;
    c.type = ACL_COMPLEX64;
    c.operation = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    c.deviceAlpha = true;
    Plan plan(MakeMatrix(c), c);
    ASSERT_EQ(plan.Analyze(true), kSuccess);
    const auto solve = plan.Solve(c.operation);
    const auto sync = aclrtSynchronizeStream(env->stream());
    ASSERT_EQ(solve, kSuccess);
    ASSERT_EQ(sync, ACL_SUCCESS);
    Verify(plan.dY.Get<float>(plan.input.size()), SparseChainGolden(plan.a, c, StoredValues(plan.a, c, false),
        Input(plan.a, c), plan.AlphaValue()), c.type);
    const auto freed = plan.workspace->FreeNow();
    EXPECT_EQ(freed, ACL_SUCCESS);
    DestroyChecked(plan);
}

class AdditionalStream {
public:
    AdditionalStream()
    {
        try {
            CheckAcl(aclrtCreateStream(&stream), "create independent stream");
            CheckSparse(aclsparseCreate(&handle), "create independent handle");
            CheckSparse(aclsparseSetStream(handle, stream), "attach independent stream");
        } catch (...) { Cleanup(); throw; }
    }
    ~AdditionalStream() { Cleanup(); }
    void Cleanup()
    {
        if (stream) aclrtSynchronizeStream(stream);
        if (handle) aclsparseDestroy(handle);
        if (stream) aclrtDestroyStream(stream);
        handle = nullptr;
        stream = nullptr;
    }
    aclrtStream stream = nullptr;
    aclsparseHandle_t handle = nullptr;
};
class RestoreFixtureHandle {
public:
    RestoreFixtureHandle() : saved(SpSVArch22Test::handle) {}
    ~RestoreFixtureHandle() { SpSVArch22Test::handle = saved; }
    aclsparseHandle_t saved;
};

TEST_F(SpSVArch22Lifetime, TwoIndependentStreamsAndBindingChangeRejection)
{
    RestoreFixtureHandle restore;
    AdditionalStream second;
    Case c1;
    Plan first(MakeMatrix(c1), c1);
    ASSERT_EQ(first.Analyze(true), kSuccess);
    Case c2;
    c2.type = ACL_COMPLEX64;
    c2.operation = ACL_SPARSE_OP_CONJUGATE_TRANSPOSE;
    c2.lower = false;
    c2.deviceAlpha = true;
    SpSVArch22Test::handle = second.handle;
    Plan other(MakeMatrix(c2), c2);
    ASSERT_EQ(other.Analyze(true), kSuccess);
    SpSVArch22Test::handle = restore.saved;
    const auto solve1 = first.Solve(c1.operation);
    SpSVArch22Test::handle = second.handle;
    const auto solve2 = other.Solve(c2.operation);
    // Both submissions precede either completion fence. Never free their
    // buffers until both streams have finished, even when a status is wrong.
    const auto sync1 = aclrtSynchronizeStream(env->stream());
    const auto sync2 = aclrtSynchronizeStream(second.stream);
    ASSERT_EQ(solve1, kSuccess);
    ASSERT_EQ(solve2, kSuccess);
    ASSERT_EQ(sync1, ACL_SUCCESS);
    ASSERT_EQ(sync2, ACL_SUCCESS);
    Verify(first.dY.Get<float>(first.input.size()), SparseChainGolden(first.a, c1, StoredValues(first.a, c1, false),
        Input(first.a, c1), first.AlphaValue()), c1.type);
    Verify(other.dY.Get<float>(other.input.size()), SparseChainGolden(other.a, c2, StoredValues(other.a, c2, false),
        Input(other.a, c2), other.AlphaValue()), c2.type);
    ExpectBitwiseFloats(first.dX.Get<float>(first.input.size()), first.input);
    ExpectBitwiseFloats(other.dX.Get<float>(other.input.size()), other.input);
    SpSVArch22Test::handle = restore.saved;
    ASSERT_EQ(aclsparseSetStream(restore.saved, second.stream), kSuccess);
    const auto wrongSolve = first.Solve(c1.operation);
    const auto wrongUpdate = aclsparseSpSV_updateMatrix(restore.saved, first.state, first.dValues.get(),
                                                        ACL_SPARSE_SPSV_UPDATE_GENERAL);
    // Restore before any fatal assertion or descriptor destruction.
    const auto restoredStream = aclsparseSetStream(restore.saved, env->stream());
    EXPECT_EQ(wrongSolve, kInvalid);
    EXPECT_EQ(wrongUpdate, kInvalid);
    ASSERT_EQ(restoredStream, kSuccess);
    DestroyChecked(first);
    DestroyChecked(other);
}

class SpSVArch22Parameters : public SpSVArch22Test {};
void VerifyBoundStrides(Plan &plan)
{
    for (int changed = 0; changed < 3; ++changed) {
        ASSERT_EQ(aclsparseCsrSetStrides(plan.mat, changed == 0 ? 2 : 1, changed == 1 ? 2 : 1,
            changed == 2 ? 2 : 1), kSuccess);
        const auto solve = plan.Solve(plan.c.operation);
        // Fence both calls before changing metadata again or releasing storage.
        const auto update = aclsparseSpSV_updateMatrix(SpSVArch22Test::handle, plan.state, plan.dValues.get(),
                                                       ACL_SPARSE_SPSV_UPDATE_GENERAL);
        const auto synced = aclrtSynchronizeStream(SpSVArch22Test::env->stream());
        EXPECT_EQ(solve, kInvalid);
        EXPECT_EQ(synced, ACL_SUCCESS);
        EXPECT_EQ(update, kInvalid);
        EXPECT_EQ(synced, ACL_SUCCESS);
        ASSERT_EQ(synced, ACL_SUCCESS);
    }
}
TEST_F(SpSVArch22Parameters, SparseStridesAndHostMatrixStorageAreRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    size_t bytes = 0;
    for (int changed = 0; changed < 3; ++changed) {
        ASSERT_EQ(aclsparseCsrSetStrides(plan.mat, changed == 0 ? 2 : 1, changed == 1 ? 2 : 1, changed == 2 ? 2 : 1),
                  kSuccess);
        const auto status = plan.Size(&bytes);
        EXPECT_EQ(status, kInvalid);
    }
    ASSERT_EQ(aclsparseCsrSetStrides(plan.mat, 1, 1, 1), kSuccess);
    {
        const auto analyzed = plan.Analyze(true);
        const auto synced = aclrtSynchronizeStream(env->stream());
        EXPECT_EQ(analyzed, kSuccess); EXPECT_EQ(synced, ACL_SUCCESS);
        ASSERT_EQ(analyzed, kSuccess); ASSERT_EQ(synced, ACL_SUCCESS);
    }
    VerifyBoundStrides(plan);
    if (::testing::Test::HasFatalFailure()) return;
    ASSERT_EQ(aclsparseCsrSetStrides(plan.mat, 1, 1, 1), kSuccess);
    for (int changed = 0; changed < 3; ++changed) {
        ASSERT_EQ(
            aclsparseCsrSetPointers(plan.mat, changed == 0 ? static_cast<void*>(plan.a.ptr.data()) : plan.dPtr.get(),
                                    changed == 1 ? static_cast<void*>(plan.a.idx.data()) : plan.dIdx.get(),
                                    changed == 2 ? static_cast<void*>(plan.initial.data()) : plan.dValues.get()),
            kSuccess);
        const auto status = plan.Analyze();
        EXPECT_EQ(status, kInvalid);
    }
}

TEST_F(SpSVArch22Parameters, RectangularMatrixAndDeviceAlphaInHostModeAreRejected)
{
    Case c;
    Plan plan(MakeMatrix(c), c);
    size_t bytes = 0;
    const auto alphaStatus = aclsparseSpSV_bufferSize(handle, c.operation, plan.dAlpha.get(), plan.mat, nullptr,
        nullptr, c.type, kDefault, plan.state, &bytes);
    EXPECT_EQ(alphaStatus, kInvalid);
    ASSERT_EQ(aclsparseDestroySpMat(plan.mat), kSuccess);
    plan.mat = nullptr;
    const auto rectangularCreate = aclsparseCreateCsr(
        &plan.mat, plan.a.m, plan.a.m - 1, plan.a.entries.size(), plan.dPtr.get(), plan.dIdx.get(), plan.dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, c.type);
    ASSERT_EQ(rectangularCreate, kSuccess);
    const auto shapeStatus = plan.Size(&bytes);
    EXPECT_EQ(shapeStatus, kInvalid);

    aclsparseSpMatDescr_t invalid = nullptr;
    // Only descriptor creation and bufferSize are probed. Every supplied
    // pointer refers to a live allocation, and no Analysis/Kernel is launched.
    const auto created = aclsparseCreateCsr(&invalid, 0, 0, 1, plan.dPtr.get(), plan.dIdx.get(), plan.dValues.get(),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, c.type);
    auto rejected = created;
    if (created == kSuccess) {
        rejected = aclsparseSpSV_bufferSize(handle, c.operation, plan.Alpha(), invalid, nullptr, nullptr,
            c.type, kDefault, plan.state, &bytes);
    }
    if (invalid != nullptr) {
        const auto destroyed = aclsparseDestroySpMat(invalid);
        invalid = nullptr;
        EXPECT_EQ(destroyed, kSuccess);
    }
    EXPECT_EQ(rejected, kInvalid);
}

std::vector<float> DrawExactMixture(uint32_t seed, int specialOffset, std::array<size_t, 3> &counts)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float special[] = {0.0f, 1.0f, -1.0f};
    std::vector<float> values;
    values.reserve(100);
    for (int scalar = 0; scalar < 100; ++scalar) {
        const int bucket = scalar < 70 ? 0 : scalar < 90 ? 1 : 2;
        ++counts[bucket];
        values.push_back(bucket == 0 ? uniform(generator) : bucket == 1 ? normal(generator) :
            special[(scalar - 90 + specialOffset) % 3]);
    }
    return values;
}
class SpSVArch22Distribution : public SpSVArch22Test {};
CombinationData DrawMixtureData(const Case &c, uint32_t seed)
{
    CombinationData data;
    auto &sampling = data.sampling;
    // Separate generators for A/X and real/imaginary. No shift, clamp,
    // rescale or rejection sampling is applied before uploading.
    const auto aReal = DrawExactMixture(seed, 0, sampling.count[0]);
    const auto aImag = c.type == ACL_COMPLEX64 ? DrawExactMixture(seed + 1, 1, sampling.count[1]) :
                                                 std::vector<float>(100, 0.0f);
    const auto xReal = DrawExactMixture(seed + 2, 2, sampling.count[2]);
    const auto xImag = c.type == ACL_COMPLEX64 ? DrawExactMixture(seed + 3, 0, sampling.count[3]) :
                                                 std::vector<float>(100, 0.0f);
    std::vector<Entry> source;
    std::vector<Complex> x;
    for (int row = 0; row < 100; ++row) {
        source.push_back({row, row, {aReal[row], aImag[row]}});
        x.emplace_back(xReal[row], xImag[row]);
    }
    data.a = EncodeEntries(100, source, c);
    data.x = std::move(x);
    return data;
}
TEST_F(SpSVArch22Distribution, ActualASourceAndXComponentsHaveExactSeventyTwentyTenMixture)
{
    size_t id = 0;
    for (auto format : {ACL_SPARSE_FORMAT_CSR, ACL_SPARSE_FORMAT_COO}) for (int type = 0; type < 2; ++type)
        for (int unit = 0; unit < 2; ++unit, ++id) {
            Case c;
            c.format = format;
            c.type = type == 0 ? ACL_FLOAT : ACL_COMPLEX64;
            c.unit = unit != 0;
            c.base = format == ACL_SPARSE_FORMAT_COO ? 1 : 0;
            c.inPlace = format == ACL_SPARSE_FORMAT_COO;
            c.deviceAlpha = unit != 0;
            c.name = CombinationKey(c, 0);
            const uint32_t seed = 0x71002a00u + static_cast<uint32_t>(id) * 4;
            const auto data = DrawMixtureData(c, seed);
            const auto &sampling = data.sampling;
            const auto &a = data.a;
            const auto &x = data.x;
            ASSERT_EQ(a.entries.size(), 100u);
            try {
                Plan plan(a, c);
                plan.input = Pack(x, c.type); plan.dX.Put(plan.input);
                ExpectBitwiseFloats(plan.dValues.Get<float>(plan.initial.size()), plan.initial);
                ExpectBitwiseFloats(plan.dX.Get<float>(plan.input.size()), plan.input);
                for (int channel = 0; channel < 4; ++channel) {
                    if ((channel == 1 || channel == 3) && c.type != ACL_COMPLEX64) continue;
                    EXPECT_EQ(sampling.count[channel][0], 70u);
                    EXPECT_EQ(sampling.count[channel][1], 20u);
                    EXPECT_EQ(sampling.count[channel][2], 10u);
                }
                CheckSparse(plan.Analyze(unit != 0), "element mixture analysis");
                VerifyRepeatedSolve(plan, SparseChainGolden(a, c, StoredValues(a, c, false), x, plan.AlphaValue()));
                CheckAcl(aclrtSynchronizeStream(env->stream()), "mixed input solve complete");
                DestroyChecked(plan);
            } catch (const std::exception &exception) { ADD_FAILURE() << exception.what(); }
            if (::testing::Test::HasFatalFailure()) return;
        }
    EXPECT_EQ(id, 8u);
}

// Validate the complete published enum domain and invalid integer sentinels.
TEST(SpSVHostCommonOnly, ValidateOperationEntirePublishedEnumDomainAndInvalidSentinels)
{
    for (int value = -128; value <= 255; ++value) {
        const auto expected = value >= 0 && value <= 2 ? kSuccess : kInvalid;
        EXPECT_EQ(spsv_common::ValidateOperation(value), expected) << "operation=" << value;
    }
    for (int value : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        EXPECT_EQ(spsv_common::ValidateOperation(value), kInvalid);
    }
}

TEST(SpSVHostCommonOnly, IsValueTypeSupportedEntirePublishedEnumDomainAndInvalidSentinels)
{
    // Probe the enclosing integer domain without requiring newer optional
    // enum symbols from a particular CANN header version.
    for (int value = -128; value <= 255; ++value) {
        const bool expected = value == static_cast<int>(ACL_FLOAT) || value == static_cast<int>(ACL_COMPLEX64);
        EXPECT_EQ(spsv_common::IsValueTypeSupported(value), expected) << "dtype=" << value;
    }
    for (int value : {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        EXPECT_FALSE(spsv_common::IsValueTypeSupported(value));
    }
}
uint64_t DoubleBits(double value) { return spsv_test::BitCopy<uint64_t>(value); }

double DoubleFromBits(uint64_t bits) { return spsv_test::BitCopy<double>(bits); }

double DoubleQuotientReference(double numerator, double denominator)
{
    if (denominator != 0.0) {
        return numerator / denominator;
    }
    if (numerator == 0.0 || std::isnan(numerator)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const bool negative = std::signbit(numerator) != std::signbit(denominator);
    const double infinity = std::numeric_limits<double>::infinity();
    return negative ? -infinity : infinity;
}

void ExpectWideResult(spsv_soft64::Real actual, double expected)
{
    if (std::isnan(expected))
        EXPECT_TRUE(spsv_soft64::IsNan(actual));
    else
        EXPECT_EQ(actual.bits, DoubleBits(expected));
}

void CheckWideArithmetic(uint64_t left, uint64_t right)
{
    const spsv_soft64::Real a{left};
    const spsv_soft64::Real b{right};
    const double x = DoubleFromBits(left);
    const double y = DoubleFromBits(right);
    ExpectWideResult(spsv_soft64::Add(a, b), x + y);
    ExpectWideResult(spsv_soft64::Sub(a, b), x - y);
    ExpectWideResult(spsv_soft64::Mul(a, b), x * y);
    ExpectWideResult(spsv_soft64::Div(a, b), DoubleQuotientReference(x, y));
}

TEST(SpSVHostCommonOnly, Binary64FallbackMatchesNativeRoundToNearestArithmetic)
{
    const uint64_t edges[] = {0,
                              0x8000000000000000ULL,
                              1,
                              0x000fffffffffffffULL,
                              0x0010000000000000ULL,
                              0x3ff0000000000000ULL,
                              0x3fefffffffffffffULL,
                              0x7fefffffffffffffULL,
                              0x7ff0000000000000ULL,
                              0xfff0000000000000ULL,
                              0x7ff8000000000001ULL};
    for (uint64_t a : edges) {
        for (uint64_t b : edges)
            CheckWideArithmetic(a, b);
    }
    std::mt19937_64 generator(0x535056524e45ULL);
    for (int sample = 0; sample < 4096; ++sample) {
        const auto a = generator(), b = generator();
        SCOPED_TRACE("arithmetic_sample=" + std::to_string(sample));
        CheckWideArithmetic(a, b);
        const uint32_t input = static_cast<uint32_t>(generator());
        const auto f = spsv_test::BitCopy<float>(input);
        auto widened = spsv_soft64::FromFloatBits(input);
        ExpectWideResult(widened, static_cast<double>(f));
        const auto narrowed = spsv_soft64::ToFloatBits(widened);
        if (std::isnan(f))
            EXPECT_EQ(narrowed & 0x7fffffffU, 0x7fc00000U);
        else
            EXPECT_EQ(narrowed, input);
    }
}

#include "spsv_buffer_span_tests.h"
#include "spsv_range_tests.h"
#include "spsv_dependency_tests.h"
#include "spsv_scaled_dependency_tests.h"
#include "spsv_duplicate_diagonal_tests.h"
} // namespace
