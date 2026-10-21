# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseSgtsv2 GTest tests
ops_sparse_add_gtest_tests(gtsv2 ${OPS_SPARSE} WARN_ON_MISSING_SRC EIGEN)

# The arch22 target runs the architecture-independent API/numerical suite from
# arch35 as a separate translation unit, plus arch22-specific tests selected by
# ops_sparse_add_gtest_tests above.  Keep the original arch35 target unchanged.
if(TARGET gtsv2_test AND "arch22" IN_LIST SOC_ARCH_DIRS)
    target_sources(gtsv2_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/gtsv2_test.cpp)
endif()
