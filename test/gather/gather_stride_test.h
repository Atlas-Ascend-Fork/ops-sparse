/**
 * Copyright (c) 2026 Xztxzt
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "test_common.h"

namespace sparse_test {

inline void CheckGatherNonUnitDenseStride(aclrtStream stream)
{
    HandleManager handle;
    handle.setStream(stream);
    std::vector<float> yHost{10.0f, 999.0f, 20.0f, 999.0f, 30.0f, 999.0f, 40.0f};
    std::vector<int32_t> indices{0, 2};
    auto dY = DeviceBuffer::copyFrom(yHost.data(), yHost.size() * sizeof(float));
    auto dIndices = DeviceBuffer::copyFrom(indices.data(), indices.size() * sizeof(int32_t));
    auto dX = DeviceBuffer::alloc(indices.size() * sizeof(float));
    auto dense = DnVecManager::create(3, dY.raw(), ACL_FLOAT);
    auto sparse = SpVecManager::create(3, 2, dIndices.raw(), dX.raw(), ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_BASE_ZERO,
                                       ACL_FLOAT);
    for (int64_t stride : {2, 3}) {
        ASSERT_EQ(aclsparseDnVecSetStride(dense.get(), stride), ACL_SPARSE_STATUS_SUCCESS);
        EXPECT_EQ(aclsparseGather(handle.get(), dense.cget(), sparse.get()), ACL_SPARSE_STATUS_NOT_SUPPORTED);
    }
    ASSERT_EQ(aclsparseDnVecSetStride(dense.get(), 1), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclsparseGather(handle.get(), dense.cget(), sparse.get()), ACL_SPARSE_STATUS_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
}

} // namespace sparse_test
