/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPMM_ARCH35_SPMM_TEST_UTILS_H_
#define TEST_SPMM_ARCH35_SPMM_TEST_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <acl/acl.h>
#include <cann_ops_sparse.h>
#include <gtest/gtest.h>
#include "securec.h"

namespace spmm_test {

class SpmmDeviceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
    }

    void TearDown() override
    {
        if (stream_ != nullptr) {
            EXPECT_EQ(aclrtDestroyStream(stream_), ACL_SUCCESS);
        }
    }

    aclrtStream stream_ = nullptr;
};

struct SpmmDeviceBuffers {
    void *rows = nullptr;
    void *columns = nullptr;
    void *values = nullptr;
    void *matB = nullptr;
    void *matC = nullptr;
    void *matOut = nullptr;
    void *workspace = nullptr;
};

inline bool SecureCopy(void *destination, size_t destinationSize,
                       const void *source, size_t count)
{
    return memcpy_s(destination, destinationSize, source, count) == EOK;
}

inline bool CheckAcl(aclError status, const char *what)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::printf("ACL_FAIL=%s STATUS=%d\n", what, static_cast<int>(status));
    return false;
}

inline bool CheckSparse(aclsparseStatus_t status, const char *what)
{
    if (status == ACL_SPARSE_STATUS_SUCCESS) {
        return true;
    }
    std::printf("SPARSE_FAIL=%s STATUS=%d\n", what, static_cast<int>(status));
    return false;
}

inline bool InitializeDevice(int32_t device, aclrtStream *stream)
{
    return CheckAcl(aclInit(nullptr), "aclInit") &&
        CheckAcl(aclrtSetDevice(device), "set device") &&
        CheckAcl(aclrtCreateStream(stream), "create stream");
}

inline float InputValue(uint64_t index, uint64_t salt, float divisor = 1024.0f)
{
    if (divisor == 0.0f) {
        std::fprintf(stderr, "InputValue divisor must be nonzero\n");
        std::abort();
    }
    uint64_t value = index + 0x9e3779b97f4a7c15ULL * salt;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<float>(static_cast<int32_t>(value % 257) - 128) / divisor;
}

inline uint16_t FloatToHalf(float value)
{
    uint32_t bits;
    if (!SecureCopy(&bits, sizeof(bits), &value, sizeof(bits))) return 0;
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t rawExp = (bits >> 23) & 0xffu;
    int32_t exp = static_cast<int32_t>(rawExp) - 127;
    uint32_t mant = bits & 0x7fffffu;
    if (rawExp == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant != 0 ? 0x200u : 0));
    }
    if (exp >= 16) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp >= -14) {
        uint32_t result = static_cast<uint32_t>(exp + 15) << 10;
        const uint32_t tail = mant & 0x1fffu;
        mant >>= 13;
        result |= mant;
        if (tail > 0x1000u || (tail == 0x1000u && (mant & 1u))) ++result;
        return static_cast<uint16_t>(sign | result);
    }
    if (exp < -25) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(-exp - 1);
    uint32_t result = mant >> shift;
    const uint32_t tail = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (tail > halfway || (tail == halfway && (result & 1u))) ++result;
    return static_cast<uint16_t>(sign | result);
}

inline float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t shift = 0;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x3ffu;
            bits = sign | ((127u - 14u - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float result;
    if (!SecureCopy(&result, sizeof(result), &bits, sizeof(result))) return 0.0f;
    return result;
}

inline uint16_t FloatToBf16(float value)
{
    uint32_t bits;
    if (!SecureCopy(&bits, sizeof(bits), &value, sizeof(bits))) return 0;
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

inline float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    if (!SecureCopy(&result, sizeof(result), &bits, sizeof(result))) return 0.0f;
    return result;
}

inline void ReleaseSpmmDeviceBuffers(SpmmDeviceBuffers *device)
{
    if (device->workspace) aclrtFree(device->workspace);
    if (device->matOut) aclrtFree(device->matOut);
    if (device->matC) aclrtFree(device->matC);
    if (device->matB) aclrtFree(device->matB);
    if (device->values) aclrtFree(device->values);
    if (device->columns) aclrtFree(device->columns);
    if (device->rows) aclrtFree(device->rows);
    *device = {};
}

}  // namespace spmm_test

#endif  // TEST_SPMM_ARCH35_SPMM_TEST_UTILS_H_
