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

#ifndef CGTSV2_NPU_WRAPPER_H_
#define CGTSV2_NPU_WRAPPER_H_

#include "descriptor_manager.h"
#include "arch35/cgtsv2_workspace.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgtsv2_test {
using Value = aclsparseComplex_t;
constexpr size_t GUARD_BYTES = 128;
constexpr unsigned char GUARD_VALUE = 0xA5;
using DispatchFlags = std::array<uint32_t, cgtsv2::kThreads>;

struct Inputs {
    std::vector<Value> dl;
    std::vector<Value> d;
    std::vector<Value> du;
    std::vector<Value> b;
};

struct GuardedDeviceBuffer {
public:
    explicit GuardedDeviceBuffer(size_t bytes)
        : backing_(sparse_test::DeviceBuffer::alloc(bytes + 2 * GUARD_BYTES)), bytes_(bytes)
    {
        std::vector<unsigned char> initial(backing_.size(), GUARD_VALUE);
        Check(aclrtMemcpy(backing_.raw(), initial.size(), initial.data(), initial.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    void* data() const { return static_cast<unsigned char*>(backing_.raw()) + GUARD_BYTES; }
    Value* values() const { return static_cast<Value*>(data()); }
    size_t size() const { return bytes_; }
    void Upload(const std::vector<Value>& values)
    {
        if (values.size() * sizeof(Value) != bytes_) {
            throw std::invalid_argument("buffer upload size");
        }
        if (bytes_ != 0) {
            Check(aclrtMemcpy(data(), bytes_, values.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
    std::vector<Value> Download() const
    {
        std::vector<Value> values(bytes_ / sizeof(Value));
        if (bytes_ != 0) {
            Check(aclrtMemcpy(values.data(), bytes_, data(), bytes_, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        return values;
    }
    void CheckGuards() const
    {
        std::array<unsigned char, GUARD_BYTES> before{};
        std::array<unsigned char, GUARD_BYTES> after{};
        Check(aclrtMemcpy(before.data(), before.size(), backing_.raw(), before.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        Check(aclrtMemcpy(after.data(), after.size(), static_cast<unsigned char*>(data()) + bytes_,
            after.size(), ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_TRUE(std::all_of(before.begin(), before.end(), [](unsigned char v) { return v == GUARD_VALUE; }));
        EXPECT_TRUE(std::all_of(after.begin(), after.end(), [](unsigned char v) { return v == GUARD_VALUE; }));
    }
private:
    static void Check(aclError status)
    {
        if (status != ACL_SUCCESS) {
            throw std::runtime_error("ACL buffer operation: " + std::to_string(status));
        }
    }
    sparse_test::DeviceBuffer backing_;
    size_t bytes_;
};

inline void ExpectUnchanged(const GuardedDeviceBuffer& buffer, const std::vector<Value>& expected)
{
    const auto actual = buffer.Download();
    ASSERT_EQ(actual.size(), expected.size());
    if (!actual.empty()) {
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Value)), 0);
    }
}

struct DeviceInputs {
    GuardedDeviceBuffer dl;
    GuardedDeviceBuffer d;
    GuardedDeviceBuffer du;
    GuardedDeviceBuffer b;

    DeviceInputs(int m, int n, int ldb, const Inputs& inputs)
        : dl(m * sizeof(Value)), d(m * sizeof(Value)), du(m * sizeof(Value)),
          b(static_cast<size_t>(ldb) * n * sizeof(Value))
    {
        dl.Upload(inputs.dl);
        d.Upload(inputs.d);
        du.Upload(inputs.du);
        b.Upload(inputs.b);
    }
    void CheckInputsAndGuards(const Inputs& inputs) const
    {
        ExpectUnchanged(dl, inputs.dl);
        ExpectUnchanged(d, inputs.d);
        ExpectUnchanged(du, inputs.du);
        dl.CheckGuards();
        d.CheckGuards();
        du.CheckGuards();
        b.CheckGuards();
    }
};

struct NpuResult {
    size_t workspaceBytes = 0;
    std::vector<std::vector<Value>> outputs;
    std::vector<DispatchFlags> dispatchFlags;
};

// Normal execution workflow; exceptional API calls remain explicit TEST_F cases.
// Keep every repeat, dispatch flag, read-only check and workspace guard observable.
inline void RunOnNpu(aclsparseHandle_t handle, int m, int n, int ldb,
    const Inputs& inputs, aclrtStream stream, NpuResult& result)
{
    DeviceInputs data(m, n, ldb, inputs);
    size_t bytes = 0;
    ASSERT_EQ(aclsparseCgtsv2_bufferSizeExt(handle, m, n, data.dl.values(), data.d.values(), data.du.values(),
        data.b.values(), ldb, &bytes), ACL_SPARSE_STATUS_SUCCESS);
    GuardedDeviceBuffer workspace(bytes);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace.data()) % cgtsv2::kAlignment, 0U);
    ASSERT_EQ(aclsparseSetStream(handle, stream), ACL_SPARSE_STATUS_SUCCESS);
    aclrtStream actualStream = nullptr;
    ASSERT_EQ(aclsparseGetStream(handle, &actualStream), ACL_SPARSE_STATUS_SUCCESS);
    ASSERT_EQ(actualStream, stream);
    result.workspaceBytes = bytes;
    for (int repeat = 0; repeat < 3; ++repeat) {
        data.b.Upload(inputs.b);
        ASSERT_EQ(aclsparseCgtsv2(handle, m, n, data.dl.values(), data.d.values(), data.du.values(), data.b.values(),
            ldb, workspace.data()), ACL_SPARSE_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        DispatchFlags flags{};
        if (n > 0) {
            ASSERT_GE(bytes, sizeof(flags));
            ASSERT_EQ(aclrtMemcpy(flags.data(), sizeof(flags), workspace.data(), sizeof(flags),
                ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
        }
        result.dispatchFlags.push_back(flags);
        result.outputs.push_back(data.b.Download());
        data.CheckInputsAndGuards(inputs);
        workspace.CheckGuards();
    }
}
} // namespace cgtsv2_test
#endif // CGTSV2_NPU_WRAPPER_H_
