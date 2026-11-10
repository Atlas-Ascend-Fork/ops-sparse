# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_sparse_add_gtest_tests(spsv ${OPS_SPARSE} EIGEN)

if(TARGET spsv_test AND "arch22" IN_LIST SOC_ARCH_DIRS)
    target_include_directories(spsv_test PRIVATE
        "${CMAKE_SOURCE_DIR}/sparse/spsv/arch22/host"
        "${CMAKE_SOURCE_DIR}/sparse/spsv/arch22/kernel"
    )
    target_sources(spsv_test PRIVATE arch22/spsv_invalid_enum_calls.c arch22/spsv_scaled_math_test.cpp)
endif()
