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
#ifndef TEST_CGTSV2_ARCH22_NPU_WRAPPER_H_
#define TEST_CGTSV2_ARCH22_NPU_WRAPPER_H_

#include "test_common.h"
#include "cgtsv2_golden.h"

namespace cgtsv2_test {
struct DeviceInput {
    sparse_test::DeviceBuffer dl;
    sparse_test::DeviceBuffer d;
    sparse_test::DeviceBuffer du;
    sparse_test::DeviceBuffer b;
    explicit DeviceInput(const Input &x)
        : dl(sparse_test::DeviceBuffer::copyFrom(x.dl.data(), x.dl.size() * sizeof(Complex))),
          d(sparse_test::DeviceBuffer::copyFrom(x.d.data(), x.d.size() * sizeof(Complex))),
          du(sparse_test::DeviceBuffer::copyFrom(x.du.data(), x.du.size() * sizeof(Complex))),
          b(sparse_test::DeviceBuffer::copyFrom(x.b.data(), x.b.size() * sizeof(Complex))) {}
};

inline aclsparseStatus_t Query(sparse_test::HandleManager &handle, const DeviceInput &d, const Input &x, size_t *bytes)
{
    return aclsparseCgtsv2_bufferSizeExt(handle.get(), x.m, x.n,
        static_cast<const Complex *>(d.dl.raw()), static_cast<const Complex *>(d.d.raw()),
        static_cast<const Complex *>(d.du.raw()), static_cast<const Complex *>(d.b.raw()), x.ldb, bytes);
}

inline aclsparseStatus_t Solve(sparse_test::HandleManager &handle, DeviceInput &d, const Input &x, void *workspace)
{
    return aclsparseCgtsv2(handle.get(), x.m, x.n,
        static_cast<const Complex *>(d.dl.raw()), static_cast<const Complex *>(d.d.raw()),
        static_cast<const Complex *>(d.du.raw()), static_cast<Complex *>(d.b.get()), x.ldb, workspace);
}

} // namespace cgtsv2_test
#endif
