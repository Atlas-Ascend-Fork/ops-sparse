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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <stdexcept>
#include "acl/acl.h"
#include "securec.h"
#include "cann_ops_sparse.h"

#define CHECK_ACL(x)                                                \
    do {                                                            \
        aclError _ret = (x);                                        \
        if (_ret != ACL_SUCCESS) {                                  \
            fprintf(stderr, "ACL error %d at %s:%d\n", _ret,       \
                    __FILE__, __LINE__);                            \
            return false;                                           \
        }                                                           \
    } while (0)

#define CHECK_ACL_SPARSE(x)                                         \
    do {                                                            \
        aclsparseStatus_t _ret = (x);                               \
        if (_ret != ACL_SPARSE_STATUS_SUCCESS) {                    \
            fprintf(stderr, "aclsparse error %d at %s:%d\n", _ret, \
                    __FILE__, __LINE__);                            \
            return false;                                           \
        }                                                           \
    } while (0)

// RAII for device buffers + sparse descriptors so CHECK_* early-returns
// still free already-allocated resources (issues #157/#161).
struct SpvvDeviceResources {
    void *dIndices = nullptr;
    void *dXValues = nullptr;
    void *dYValues = nullptr;
    void *dResult = nullptr;
    aclsparseHandle_t handle = nullptr;
    aclsparseSpVecDescr_t spVecX = nullptr;
    aclsparseDnVecDescr_t dnVecY = nullptr;

    ~SpvvDeviceResources()
    {
        if (spVecX != nullptr) {
            aclsparseDestroySpVec(spVecX);
        }
        if (dnVecY != nullptr) {
            aclsparseDestroyDnVec(dnVecY);
        }
        if (handle != nullptr) {
            aclsparseDestroy(handle);
        }
        if (dIndices != nullptr) {
            aclrtFree(dIndices);
        }
        if (dXValues != nullptr) {
            aclrtFree(dXValues);
        }
        if (dYValues != nullptr) {
            aclrtFree(dYValues);
        }
        if (dResult != nullptr) {
            aclrtFree(dResult);
        }
    }

    SpvvDeviceResources() = default;
    SpvvDeviceResources(const SpvvDeviceResources &) = delete;
    SpvvDeviceResources &operator=(const SpvvDeviceResources &) = delete;
};

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

static void generate_random_spvv_data(
    std::vector<int32_t> &indices, std::vector<float> &xValues,
    std::vector<float> &yValues, uint32_t nnz, uint32_t yLen)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> valDist(-1.0f, 1.0f);

    // Generate unique sorted indices via Fisher-Yates shuffle + take first nnz
    indices.resize(nnz);
    if (nnz < yLen) {
        std::vector<int32_t> pool(yLen);
        std::iota(pool.begin(), pool.end(), 0);
        for (uint32_t i = 0; i < nnz; i++) {
            std::uniform_int_distribution<int32_t> swapDist(i, static_cast<int32_t>(yLen) - 1);
            std::swap(pool[i], pool[swapDist(rng)]);
            indices[i] = pool[i];
        }
        std::sort(indices.begin(), indices.end());
    } else {
        // nnz >= yLen: use all indices, sorted
        std::iota(indices.begin(), indices.end(), 0);
    }

    xValues.resize(nnz);
    for (uint32_t i = 0; i < nnz; i++) {
        xValues[i] = valDist(rng);
    }

    yValues.resize(yLen);
    for (uint32_t i = 0; i < yLen; i++) {
        yValues[i] = valDist(rng);
    }
}

static float compute_golden_fp32(const std::vector<int32_t> &indices,
    const std::vector<float> &xValues, const std::vector<float> &yValues)
{
    double sum = 0.0;
    for (size_t i = 0; i < indices.size(); i++) {
        sum += (double)xValues[i] * (double)yValues[indices[i]];
    }
    return (float)sum;
}

// Software FP16 <-> FP32 conversion (no dependency on AscendC half type)

static inline void checked_memcpy(void *dst, size_t dstSize, const void *src, size_t count)
{
    const errno_t status = memcpy_s(dst, dstSize, src, count);
    if (status != EOK) {
        throw std::runtime_error("SPVV host memcpy_s failed: " + std::to_string(status));
    }
}

static inline uint16_t fp32_bits_to_fp16_bits(uint32_t x)
{
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x007FFFFF;

    if (exp >= 16) {
        // Keep NaN handling on the existing overflow path so ordinary finite
        // conversions incur no additional branch. Preserve sign and payload.
        uint32_t payload = 0;
        if ((x & 0x7F800000) == 0x7F800000 && mant != 0) {
            payload = mant >> 13;
            if (payload == 0) {
                payload = 1;
            }
        }
        return static_cast<uint16_t>(sign | 0x7C00 | payload);
    }

    if (exp >= -14) {
        // Bias by half an ulp minus one, then use the retained low bit for ties
        // to even. Addition also carries a rounded mantissa into the exponent.
        uint32_t roundedMant = (mant + 0xFFF + ((mant >> 13) & 1u)) >> 13;
        uint32_t result = (static_cast<uint32_t>(exp + 15) << 10) + roundedMant;
        return static_cast<uint16_t>(sign | result);
    }

    if (exp >= -25) {
        uint32_t significand = 0x00800000 | mant;
        uint32_t shift = static_cast<uint32_t>(-exp - 1);
        uint32_t halfway = 1u << (shift - 1u);
        uint32_t fraction = (significand + halfway - 1u + ((significand >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | fraction);
    }

    return static_cast<uint16_t>(sign);
}

static uint16_t fp32_to_fp16_bits(float f)
{
    uint32_t bits;
    checked_memcpy(&bits, sizeof(bits), &f, sizeof(bits));
    return fp32_bits_to_fp16_bits(bits);
}

static float fp16_bits_to_fp32(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            int32_t exp32 = 113;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp32--;
            }
            mant &= 0x3FF;
            f = sign | (static_cast<uint32_t>(exp32) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    checked_memcpy(&result, sizeof(result), &f, sizeof(result));
    return result;
}

static bool test_fp16_nonfinite_helpers()
{
    struct NonfiniteCase {
        uint32_t input;
        uint16_t expected;
    };
    const NonfiniteCase cases[] = {
        {0x7F800000, 0x7C00}, {0xFF800000, 0xFC00}, // infinities
        {0x7FC00001, 0x7E00}, {0xFFC00001, 0xFE00}, // signed quiet NaNs
        {0x7F802000, 0x7C01}, {0xFF802000, 0xFC01}, // signed signaling NaNs
        {0x7F800001, 0x7C01}, {0xFF800001, 0xFC01}, // payload lost by truncation
        {0x7FFFFFFF, 0x7FFF}, {0xFFFFFFFF, 0xFFFF}, // maximum payload
    };
    for (const NonfiniteCase &testCase : cases) {
        float input;
        checked_memcpy(&input, sizeof(input), &testCase.input, sizeof(input));
        uint16_t actual = fp32_to_fp16_bits(input);
        float decoded = fp16_bits_to_fp32(testCase.expected);
        uint32_t decodedBits;
        checked_memcpy(&decodedBits, sizeof(decodedBits), &decoded, sizeof(decodedBits));
        const uint32_t expectedBits = (static_cast<uint32_t>(testCase.expected & 0x8000) << 16) |
                                     0x7F800000 | ((testCase.expected & 0x3FF) << 13);
        if (actual != testCase.expected || decodedBits != expectedBits ||
            fp32_to_fp16_bits(decoded) != testCase.expected) {
            fprintf(stderr, "FP16 nonfinite mismatch: input=0x%08x expected=0x%04x actual=0x%04x\n",
                    testCase.input, static_cast<unsigned>(testCase.expected), static_cast<unsigned>(actual));
            return false;
        }
    }
    return true;
}

static bool test_fp16_conversion_helpers()
{
    if (!test_fp16_nonfinite_helpers()) {
        return false;
    }
    struct ConversionCase {
        float input;
        uint16_t expected;
    };
    const ConversionCase cases[] = {
        {0x1p-24f, 0x0001},
        {0x1p-20f, 0x0010},
        {0x1p-14f, 0x0400},
        {1.00048828125f, 0x3C00},
        {1.0009f, 0x3C01},
        {1.00146484375f, 0x3C02},
        {-0x1p-20f, 0x8010},
    };
    for (const ConversionCase &testCase : cases) {
        uint16_t actual = fp32_to_fp16_bits(testCase.input);
        if (actual != testCase.expected) {
            fprintf(stderr, "FP16 conversion mismatch: input=%.9g expected=0x%04x actual=0x%04x\n",
                    testCase.input, static_cast<unsigned>(testCase.expected),
                    static_cast<unsigned>(actual));
            return false;
        }
    }

    if (fp16_bits_to_fp32(0x0010) != 0x1p-20f) {
        fprintf(stderr, "FP16 subnormal decode mismatch: bits=0x0010\n");
        return false;
    }
    for (uint32_t bits = 1; bits <= 0x03FF; bits++) {
        uint16_t positive = static_cast<uint16_t>(bits);
        uint16_t negative = static_cast<uint16_t>(bits | 0x8000);
        float expected = static_cast<float>(bits) * 0x1p-24f;
        float positiveValue = fp16_bits_to_fp32(positive);
        float negativeValue = fp16_bits_to_fp32(negative);
        if (positiveValue != expected || negativeValue != -expected ||
            fp32_to_fp16_bits(positiveValue) != positive ||
            fp32_to_fp16_bits(negativeValue) != negative) {
            fprintf(stderr, "FP16 subnormal round-trip mismatch: bits=0x%04x\n", bits);
            return false;
        }
    }

    printf("FP16 conversion helper boundaries: PASS\n");
    return true;
}

static float compute_golden_fp16(const std::vector<int32_t> &indices,
    const std::vector<uint16_t> &xValuesFp16, const std::vector<uint16_t> &yValuesFp16)
{
    double sum = 0.0;
    for (size_t i = 0; i < indices.size(); i++) {
        double xv = fp16_bits_to_fp32(xValuesFp16[i]);
        double yv = fp16_bits_to_fp32(yValuesFp16[indices[i]]);
        sum += xv * yv;
    }
    return (float)sum;
}

static void float_to_fp16(const std::vector<float> &src, std::vector<uint16_t> &dst)
{
    dst.resize(src.size());
    // Amortize the secure-copy call across a block while retaining the exact
    // scalar integer conversion, including NaN payloads and subnormals.
    constexpr size_t kBlockSize = 8;
    size_t i = 0;
    for (; src.size() - i >= kBlockSize; i += kBlockSize) {
        uint32_t bits[kBlockSize];
        checked_memcpy(bits, sizeof(bits), src.data() + i, sizeof(bits));
        for (size_t lane = 0; lane < kBlockSize; ++lane) {
            dst[i + lane] = fp32_bits_to_fp16_bits(bits[lane]);
        }
    }
    for (; i < src.size(); ++i) {
        dst[i] = fp32_to_fp16_bits(src[i]);
    }
}

// ---------------------------------------------------------------------------
// Helper: run SpVV and sync
// ---------------------------------------------------------------------------

static aclsparseStatus_t run_spvv(aclsparseHandle_t handle,
    aclsparseOperation_t op, aclsparseConstSpVecDescr_t x,
    aclsparseConstDnVecDescr_t y, void *dResult,
    aclDataType computeType)
{
    aclsparseStatus_t st = aclsparseSpvv(handle, op, x, y, dResult, computeType);
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        return st;
    }

    // Synchronize to ensure kernel completes before reading result
    aclrtStream stream = nullptr;
    aclsparseGetStream(handle, &stream);
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d at %s:%d\n", aclRet, __FILE__, __LINE__);
        return ACL_SPARSE_STATUS_EXECUTION_FAILED;
    }

    return ACL_SPARSE_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Dtype filter: selectable via argv[1] = "fp32" | "fp16" | "all"(default).
// Lets FP32 and FP16 paths be exercised independently to isolate dtype-
// specific issues (e.g. SyncAll deadlocks that only manifest on one path).
// ---------------------------------------------------------------------------

enum DtypeFilter { DT_ALL = 0, DT_FP32 = 1, DT_FP16 = 2 };
static DtypeFilter g_dtypeFilter = DT_ALL;

static bool dtype_enabled(aclDataType dtype)
{
    if (g_dtypeFilter == DT_ALL) return true;
    if (g_dtypeFilter == DT_FP32) return (dtype == ACL_FLOAT);
    return (dtype == ACL_FLOAT16);
}

// ---------------------------------------------------------------------------
// Test case table — shared between FP32 and FP16, covering scale & sparsity.
// ---------------------------------------------------------------------------

struct SpvvCase {
    uint32_t nnz;
    uint32_t yLen;
    const char *tag;
};

// Correctness cases: edge, sparse, small, medium, large.
static const SpvvCase kCorrectnessCases[] = {
    {1,     256,      "tiny"        },
    {0,     256,      "zero-nnz"    },
    {100,   1000000,  "sparse-1M"   },  // density 1e-4, scalar fallback
    {100,   10000000, "sparse-10M"  },  // density 1e-5, heavy scalar fallback
    {256,   10240,    "small"       },
    {1024,  10240,    "small-1K"    },
    {4096,  10240,    "small-4K"    },
    {10000, 102400,   "medium"      },
    {100000,1000000,  "large-1M"    },
    {100000,10000000, "large-10M"   },
};

// ---------------------------------------------------------------------------
// Unified correctness runner (FP32 / FP16)
// ---------------------------------------------------------------------------

// Run one SpVV correctness case. Assumes aclrtSetDevice already done; does not
// call aclInit/aclFinalize (caller manages runtime lifecycle).
static bool run_spvv_once(aclDataType dtype, uint32_t nnz, uint32_t yLen, const char *tag)
{
    bool isFp16 = (dtype == ACL_FLOAT16);
    const char *dtName = isFp16 ? "FP16" : "FP32";

    std::vector<int32_t> indices;
    std::vector<float> xValuesFp32, yValuesFp32;
    generate_random_spvv_data(indices, xValuesFp32, yValuesFp32, nnz, yLen);

    std::vector<uint16_t> xValuesFp16, yValuesFp16;
    float golden;
    if (isFp16) {
        float_to_fp16(xValuesFp32, xValuesFp16);
        float_to_fp16(yValuesFp32, yValuesFp16);
        golden = compute_golden_fp16(indices, xValuesFp16, yValuesFp16);
    } else {
        golden = compute_golden_fp32(indices, xValuesFp32, yValuesFp32);
    }

    size_t elemSz = isFp16 ? sizeof(uint16_t) : sizeof(float);
    SpvvDeviceResources res;
    CHECK_ACL(aclrtMalloc(&res.dIndices, std::max<size_t>(nnz * sizeof(int32_t), 4), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&res.dXValues, std::max<size_t>(nnz * elemSz, 4), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&res.dYValues, yLen * elemSz, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&res.dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

    if (nnz > 0) {
        CHECK_ACL(aclrtMemcpy(res.dIndices, nnz * sizeof(int32_t), indices.data(),
            nnz * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
        const void *xv = isFp16 ? (const void *)xValuesFp16.data() : (const void *)xValuesFp32.data();
        CHECK_ACL(aclrtMemcpy(res.dXValues, nnz * elemSz, xv, nnz * elemSz, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    const void *yv = isFp16 ? (const void *)yValuesFp16.data() : (const void *)yValuesFp32.data();
    CHECK_ACL(aclrtMemcpy(res.dYValues, yLen * elemSz, yv, yLen * elemSz, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL_SPARSE(aclsparseCreate(&res.handle));
    CHECK_ACL_SPARSE(aclsparseCreateSpVec(&res.spVecX, yLen, nnz, res.dIndices, res.dXValues,
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype));
    CHECK_ACL_SPARSE(aclsparseCreateDnVec(&res.dnVecY, yLen, res.dYValues, dtype));

    aclsparseStatus_t st = run_spvv(res.handle, ACL_SPARSE_OP_NON_TRANSPOSE,
        res.spVecX, res.dnVecY, res.dResult, ACL_FLOAT);
    bool pass = true;
    if (st != ACL_SPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "aclsparseSpvv %s [%s] failed: %d\n", dtName, tag, st);
        pass = false;
    } else {
        float hResult = 0.0f;
        CHECK_ACL(aclrtMemcpy(&hResult, sizeof(float), res.dResult, sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST));
        float tol = (isFp16 ? 1e-2f : 1e-3f) * std::max(1.0f, std::abs(golden));
        pass = std::abs(hResult - golden) < tol;
        printf("  %s [%s] nnz=%u yLen=%u: golden=%.6f npu=%.6f diff=%.2e → %s\n",
               dtName, tag, nnz, yLen, golden, hResult, std::abs(hResult - golden),
               pass ? "PASS" : "FAIL");
    }

    return pass;
}

static bool test_spvv_case(aclDataType dtype, uint32_t nnz, uint32_t yLen, const char *tag)
{
    bool isFp16 = (dtype == ACL_FLOAT16);
    const char *dtName = isFp16 ? "FP16" : "FP32";
    printf("=== %s SpVV [%s]: nnz=%u, yLen=%u ===\n", dtName, tag, nnz, yLen);

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    bool pass = run_spvv_once(dtype, nnz, yLen, tag);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return pass;
}

// ---------------------------------------------------------------------------
// Randomized fuzz test: 100 iterations, random nnz/yLen, FP32 + FP16.
// ---------------------------------------------------------------------------

static bool test_spvv_random()
{
    const int N_ITERS = 100;
    printf("=== Random fuzz: %d iters × {FP32, FP16} ===\n", N_ITERS);

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    std::mt19937 rng(20260626);
    int passed = 0, failed = 0;

    for (int it = 0; it < N_ITERS; it++) {
        // yLen in [256, 2M]; nnz in [0, yLen] (nnz <= yLen required).
        std::uniform_int_distribution<uint32_t> yLenDist(256, 2000000);
        uint32_t yLen = yLenDist(rng);
        std::uniform_int_distribution<uint32_t> nnzDist(0, yLen);
        uint32_t nnz = nnzDist(rng);

        std::ostringstream tagStream;
        tagStream << "rand#" << it;
        std::string tag = tagStream.str();

        if (dtype_enabled(ACL_FLOAT)) {
            run_spvv_once(ACL_FLOAT, nnz, yLen, tag.c_str()) ? passed++ : failed++;
        }
        if (dtype_enabled(ACL_FLOAT16)) {
            run_spvv_once(ACL_FLOAT16, nnz, yLen, tag.c_str()) ? passed++ : failed++;
        }
    }

    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    printf("  Random fuzz: %d passed, %d failed\n", passed, failed);
    return failed == 0;
}

// ---------------------------------------------------------------------------
// Benchmark: FP32 & FP16 across scales / sparsities (warmup=5, iters=20)
// ---------------------------------------------------------------------------

static const SpvvCase kBenchCases[] = {
    // Scalar-fallback path (very sparse: < ALIGN_ELEM indices per y segment)
    {100,   1000000,  "scalar-1M"    },  // density 1e-4
    {100,   10000000, "scalar-10M"   },  // density 1e-5, heavy
    {1000,  10000000, "scalar-1K10M" },  // density 1e-4
    // Small nnz
    {1,     256,      "tiny"         },
    {256,   10240,    "small-256"    },
    {1024,  10240,    "small-1K"     },
    {4096,  10240,    "small-4K"     },
    // Medium / large
    {10000, 102400,   "medium"       },
    {100000,1000000,  "large-1M"     },
    {100000,10000000, "large-10M"    },
};

static bool test_spvv_benchmark()
{
    static const aclDataType kDtypes[] = {ACL_FLOAT, ACL_FLOAT16};
    bool allPass = true;

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    int64_t coreNum = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
    printf("=== Benchmark: FP32 & FP16 (warmup=5, iters=20), cores=%ld ===\n", coreNum);

    int nCases = sizeof(kBenchCases) / sizeof(kBenchCases[0]);
    for (size_t d = 0; d < sizeof(kDtypes) / sizeof(kDtypes[0]); d++) {
        aclDataType dtype = kDtypes[d];
        if (!dtype_enabled(dtype)) continue;
        bool isFp16 = (dtype == ACL_FLOAT16);
        const char *dtName = isFp16 ? "FP16" : "FP32";
        size_t elemSz = isFp16 ? sizeof(uint16_t) : sizeof(float);

        for (int ci = 0; ci < nCases; ci++) {
            uint32_t nnz = kBenchCases[ci].nnz;
            uint32_t yLen = kBenchCases[ci].yLen;
            printf("\n--- %s [%s] nnz=%u yLen=%u ---\n", dtName, kBenchCases[ci].tag, nnz, yLen);

            std::vector<int32_t> indices;
            std::vector<float> xValuesFp32, yValuesFp32;
            generate_random_spvv_data(indices, xValuesFp32, yValuesFp32, nnz, yLen);
            std::vector<uint16_t> xValuesFp16, yValuesFp16;
            float golden;
            if (isFp16) {
                float_to_fp16(xValuesFp32, xValuesFp16);
                float_to_fp16(yValuesFp32, yValuesFp16);
                golden = compute_golden_fp16(indices, xValuesFp16, yValuesFp16);
            } else {
                golden = compute_golden_fp32(indices, xValuesFp32, yValuesFp32);
            }

            SpvvDeviceResources res;
            CHECK_ACL(aclrtMalloc(&res.dIndices, std::max<size_t>(nnz * sizeof(int32_t), 4), ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK_ACL(aclrtMalloc(&res.dXValues, std::max<size_t>(nnz * elemSz, 4), ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK_ACL(aclrtMalloc(&res.dYValues, yLen * elemSz, ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK_ACL(aclrtMalloc(&res.dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

            if (nnz > 0) {
                CHECK_ACL(aclrtMemcpy(res.dIndices, nnz * sizeof(int32_t), indices.data(),
                    nnz * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
                const void *xv = isFp16 ? (const void *)xValuesFp16.data() : (const void *)xValuesFp32.data();
                CHECK_ACL(aclrtMemcpy(res.dXValues, nnz * elemSz, xv, nnz * elemSz, ACL_MEMCPY_HOST_TO_DEVICE));
            }
            const void *yv = isFp16 ? (const void *)yValuesFp16.data() : (const void *)yValuesFp32.data();
            CHECK_ACL(aclrtMemcpy(res.dYValues, yLen * elemSz, yv, yLen * elemSz, ACL_MEMCPY_HOST_TO_DEVICE));

            CHECK_ACL_SPARSE(aclsparseCreate(&res.handle));
            CHECK_ACL_SPARSE(aclsparseCreateSpVec(&res.spVecX, yLen, nnz, res.dIndices, res.dXValues,
                ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO, dtype));
            CHECK_ACL_SPARSE(aclsparseCreateDnVec(&res.dnVecY, yLen, res.dYValues, dtype));

            for (int w = 0; w < 5; w++) {
                aclsparseSpvv(res.handle, ACL_SPARSE_OP_NON_TRANSPOSE, res.spVecX, res.dnVecY, res.dResult, ACL_FLOAT);
            }
            CHECK_ACL(aclrtSynchronizeDevice());

            const int N_ITERS = 20;
            double sumMs = 0.0, sumSq = 0.0;
            float minMs = 1e30f, maxMs = 0.0f;
            aclrtEvent startEvt, stopEvt;
            aclrtCreateEvent(&startEvt);
            aclrtCreateEvent(&stopEvt);
            for (int iter = 0; iter < N_ITERS; iter++) {
                aclrtRecordEvent(startEvt, nullptr);
                aclsparseSpvv(res.handle, ACL_SPARSE_OP_NON_TRANSPOSE, res.spVecX, res.dnVecY, res.dResult, ACL_FLOAT);
                aclrtRecordEvent(stopEvt, nullptr);
                aclrtSynchronizeEvent(stopEvt);
                float ms = 0.0f;
                aclrtEventElapsedTime(&ms, startEvt, stopEvt);
                sumMs += ms;
                sumSq += (double)ms * ms;
                if (ms < minMs) minMs = ms;
                if (ms > maxMs) maxMs = ms;
            }
            aclrtDestroyEvent(startEvt);
            aclrtDestroyEvent(stopEvt);

            double avgMs = sumMs / N_ITERS;
            double stdMs = sqrt(sumSq / N_ITERS - avgMs * avgMs);
            double throughput = (nnz / avgMs) / 1000.0;  // Mnnz/s

            CHECK_ACL(aclrtSynchronizeDevice());
            float hResult = 0.0f;
            CHECK_ACL(aclrtMemcpy(&hResult, sizeof(float), res.dResult, sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST));
            float tol = (isFp16 ? 1e-2f : 1e-3f) * std::max(1.0f, std::abs(golden));
            bool pass = std::abs(hResult - golden) < tol;

            printf("  avg=%.4f ms, min=%.4f ms, max=%.4f ms, %.1f Mnnz/s | diff=%.2e → %s\n",
                   avgMs, minMs, maxMs, throughput, std::abs(hResult - golden), pass ? "PASS" : "FAIL");
            if (!pass) allPass = false;
        }
    }

    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return allPass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static int run_spvv_tests(int argc, char *argv[])
{
    // Optional argv[1]: "fp32" | "fp16" | "all" (default). Selects which dtype
    // paths to exercise so FP32/FP16 can be validated independently.
    if (argc > 1) {
        std::string sel = argv[1];
        if (sel == "fp32") g_dtypeFilter = DT_FP32;
        else if (sel == "fp16") g_dtypeFilter = DT_FP16;
        else if (sel == "all") g_dtypeFilter = DT_ALL;
        else {
            fprintf(stderr, "Usage: %s [fp32|fp16|all]\n", argv[0]);
            return 1;
        }
    }
    printf("Dtype filter: %s\n",
        g_dtypeFilter == DT_FP32 ? "FP32" : g_dtypeFilter == DT_FP16 ? "FP16" : "ALL");

    if (g_dtypeFilter != DT_FP32 && !test_fp16_conversion_helpers()) {
        return 1;
    }

    int passed = 0, failed = 0;

    // Correctness: each case × {FP32, FP16}
    for (size_t i = 0; i < sizeof(kCorrectnessCases) / sizeof(kCorrectnessCases[0]); i++) {
        const SpvvCase &c = kCorrectnessCases[i];
        if (dtype_enabled(ACL_FLOAT)) {
            test_spvv_case(ACL_FLOAT, c.nnz, c.yLen, c.tag) ? passed++ : failed++;
        }
        if (dtype_enabled(ACL_FLOAT16)) {
            test_spvv_case(ACL_FLOAT16, c.nnz, c.yLen, c.tag) ? passed++ : failed++;
        }
    }

    // Benchmark: FP32 & FP16 across scales / sparsities
    test_spvv_benchmark() ? passed++ : failed++;

    // Randomized fuzz: 100 iters × {FP32, FP16}
    test_spvv_random() ? passed++ : failed++;

    printf("\n========================================\n");
    printf("Total: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}

int main(int argc, char *argv[])
{
    try {
        return run_spvv_tests(argc, argv);
    } catch (const std::exception &error) {
        fprintf(stderr, "SPVV test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
