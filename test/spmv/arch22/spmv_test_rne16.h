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
#ifndef SPMV_TEST_RNE16_H
#define SPMV_TEST_RNE16_H

// Exact products put the compensated sum just below/at/above a 16-bit midpoint.
// Expected values are representable endpoints, never a double-to-float conversion.
template <typename T>
struct SpmvRne16Input {
    std::vector<int32_t> rows{0}, cols;
    std::vector<T> values, x, expected;
};

template <typename T>
static SpmvRne16Input<T> BuildSpmvRne16Input() {
    constexpr bool isHalf = std::is_same_v<T, half>;
    const float ulp = isHalf ? 0x1p-10f : 0x1p-7f;
    const float tinyA = isHalf ? 0x1p-14f : 0x1p-12f;
    const float tinyB = isHalf ? 0x1p-11f : 0x1p-13f;
    SpmvRne16Input<T> input;
    input.x = {static_cast<T>(1.f), static_cast<T>(1.f), static_cast<T>(tinyB)};
    for (float scale : {1.f, 256.f}) {
        for (float sign : {1.f, -1.f}) {
            for (int parity = 0; parity < 2; ++parity) {
                const float lower = (1.f + parity * ulp) * scale;
                for (int delta : {-1, 0, 1}) {
                    input.cols.insert(input.cols.end(), {0, 1, 2});
                    input.values.insert(input.values.end(), {static_cast<T>(sign * lower),
                        static_cast<T>(sign * ulp * scale * .5f), static_cast<T>(sign * delta * tinyA * scale)});
                    input.rows.push_back(static_cast<int32_t>(input.cols.size()));
                    const bool upper = delta > 0 || (delta == 0 && parity == 1);
                    input.expected.push_back(static_cast<T>(sign * (lower + (upper ? ulp * scale : 0.f))));
                }
            }
        }
    }
    return input;
}

template <typename T>
static uint16_t SpmvRne16Bits(const T &value) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

template <typename T>
static int CheckSpmvRne16(const std::vector<T> &expected, const std::vector<T> &actual,
                          bool transpose, float beta) {
    LOG_PRINT("[RNE16] transpose=%d beta=%.0f\n", transpose, beta);
    int failures = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto want = SpmvRne16Bits(expected[i]);
        const auto got = SpmvRne16Bits(actual[i]);
        if (want != got) {
            LOG_PRINT("[ERROR] RNE16 %s row=%zu expected=0x%04x actual=0x%04x\n",
                (std::is_same_v<T, half> ? "half" : "bf16"), i, want, got);
            ++failures;
        }
    }
    LOG_PRINT("[RNE16] %s failures=%d/%zu\n", (std::is_same_v<T, half> ? "half" : "bf16"),
        failures, expected.size());
    return failures == 0 ? 0 : 1;
}

template <typename T>
static int ExecuteSpmvRne16(SpmvRunEnv &env, aclrtStream stream, std::vector<T> &actual,
                            bool transpose, float beta, float alpha = 1.f) {
    const auto op = transpose ? ACL_SPARSE_OP_TRANSPOSE : ACL_SPARSE_OP_NON_TRANSPOSE;
    size_t bytes = 0;
    auto st = aclsparseSpMVGetBufferSize(env.spHandle, op, &alpha,
        env.matDesc, env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, &bytes);
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    std::unique_ptr<void, aclError (*)(void *)> workspace(nullptr, aclrtFree);
    if (bytes > 0) {
        void *ptr = nullptr;
        int ret = aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        workspace.reset(ptr);
    }
    st = aclsparseSpMVPreprocess(env.spHandle, op, &alpha, env.matDesc,
        env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.get());
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    st = aclsparseSpMV(env.spHandle, op, &alpha, env.matDesc,
        env.vecXDesc, &beta, env.vecYDesc, ACL_FLOAT, ACL_SPARSE_SPMV_ALG_DEFAULT, workspace.get());
    CHECK_RET(st == ACL_SPARSE_STATUS_SUCCESS, return st);
    int ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return aclrtMemcpy(actual.data(), actual.size() * sizeof(T), env.yDevice,
        actual.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
}

template <typename T>
static int RunSpmvRne16Input(SpmvRne16Input<T> &input, const std::vector<T> &y,
                            aclrtStream stream, bool transpose, float beta, float alpha = 1.f) {
    const size_t count = input.expected.size();
    std::vector<T> actual(count);
    SpmvRunEnv env;
    int ret = SetupSpmvRunEnv<T, T>(stream, transpose ? input.x.size() : count,
        transpose ? count : input.x.size(), input.rows, input.cols, input.values,
        input.x, y, 0, AclTypeOf<T>(), AclTypeOf<T>(), env);
    if (ret == 0) {
        ret = ExecuteSpmvRne16(env, stream, actual, transpose, beta, alpha);
    }
    if (env.matDesc != nullptr) aclsparseDestroySpMat(env.matDesc);
    if (env.vecXDesc != nullptr) aclsparseDestroyDnVec(env.vecXDesc);
    if (env.vecYDesc != nullptr) aclsparseDestroyDnVec(env.vecYDesc);
    if (env.spHandle != nullptr) aclsparseDestroy(env.spHandle);
    CHECK_RET(ret == 0, return ret);
    return CheckSpmvRne16(input.expected, actual, transpose, beta);
}

template <typename T>
static int TestSpmvRne16(aclrtStream stream, bool transpose = false, float beta = 0.f) {
    auto input = BuildSpmvRne16Input<T>();
    const size_t count = input.expected.size();
    std::vector<T> y(count, static_cast<T>(0.f));
    if (beta != 0.f) {
        for (size_t r = 0; r < count; ++r) {
            y[r] = input.values[3 * r];
            input.values[3 * r] = static_cast<T>(0.f);
        }
    }
    if (transpose) {
        std::vector<T> transposed;
        input.rows = {0, static_cast<int32_t>(count), static_cast<int32_t>(2 * count),
            static_cast<int32_t>(3 * count)};
        input.cols.clear();
        for (size_t c = 0; c < 3; ++c) {
            for (size_t r = 0; r < count; ++r) {
                input.cols.push_back(static_cast<int32_t>(r));
                transposed.push_back(input.values[3 * r + c]);
            }
        }
        input.values = std::move(transposed);
    }
    return RunSpmvRne16Input(input, y, stream, transpose, beta);
}

static int TestSpmvRne16Large(aclrtStream stream, bool transpose, float sign, int delta) {
    SpmvRne16Input<bfloat16_t> input;
    input.rows = {0, 1};
    input.cols = {0};
    input.values = {bfloat16_t(sign * 0x1p127f)};
    input.x = {bfloat16_t(1.f)};
    // Exact sum: sign * (2^127 + 2^119 + delta * 2^95).
    // delta == 0 is the midpoint with an even lower endpoint.
    input.expected = {bfloat16_t(sign * (delta > 0 ? 0x1.02p127f : 0x1p127f))};
    const std::vector<bfloat16_t> y{bfloat16_t(1.f)};
    const float beta = sign * delta * 0x1p95f;
    LOG_PRINT("[RNE16 large] sign=%.0f delta=%d\n", sign, delta);
    return RunSpmvRne16Input(input, y, stream, transpose, beta, 0x1.01p0f);
}

static void RecordSpmvRne16Result(TestStats &stats, int ret) {
    ++stats.total;
    if (ret == 0) {
        ++stats.passed;
    } else {
        ++stats.failed;
        stats.failedCases.push_back("[rne16] exact midpoint rounding mismatch");
    }
}

static void RunSpmvRne16Regression(TestStats &stats, aclrtStream stream) {
    for (bool transpose : {false, true}) {
        for (float beta : {0.f, 1.f}) {
            RecordSpmvRne16Result(stats, TestSpmvRne16<half>(stream, transpose, beta));
            RecordSpmvRne16Result(stats, TestSpmvRne16<bfloat16_t>(stream, transpose, beta));
        }
        for (float sign : {1.f, -1.f}) {
            for (int delta : {-1, 0, 1}) {
                RecordSpmvRne16Result(stats, TestSpmvRne16Large(stream, transpose, sign, delta));
            }
        }
    }
}
#endif
