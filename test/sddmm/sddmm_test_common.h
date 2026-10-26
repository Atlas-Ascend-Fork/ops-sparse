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

#ifndef TEST_SDDMM_SDDMM_TEST_COMMON_H_
#define TEST_SDDMM_SDDMM_TEST_COMMON_H_

#include "sddmm_test_layout.h"

namespace sparse_test {
using SddmmParameterizedTest = SddmmFunctionalTestFixture;

inline void VerifySddmmPrecision(const std::string& caseName, const SddmmNpuResult& result,
                                 const std::vector<double>& expected, aclDataType dtype, int64_t nnz)
{
    if (nnz > 0) {
        EXPECT_TRUE(VerifySddmmPrecisionCore(caseName, result, expected, dtype, nnz))
            << "MIXED_TOLERANCE verification failed for " << caseName;
    }
}
} // namespace sparse_test

#endif
