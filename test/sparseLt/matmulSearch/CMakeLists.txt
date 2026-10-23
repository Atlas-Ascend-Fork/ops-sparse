# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseLtMatmulSearch GTest tests（Host API auto-tuning，NO_SRC_DIR 范式——
# 纯 host 算子，源文件在叶子目录顶层；布局对齐 matmulPlanInit 先例）。
#
# 交付件（PR 范围 = ST 8 文件）：
#   matmulSearch_param.h / matmulSearch_golden.h / matmulSearch_npu_wrapper.h
#   matmulSearch_timing.h /
#   matmulSearch_test.cpp + matmulSearch_test.csv
#   （L0+L1，88 CSV 用例；L2 异常 X01~X50 以 TEST_F 实现（43 条）；合计 131）
#
# 依赖（EXTRA_INCLUDES 复用 ltmatmul 测试资产与 sparseLt 白盒 internal 头，不复制）。
# aclsparseLtMatmulSearch 声明位于公开头 include/cann_ops_sparseLt.h。

if(NOT ENABLE_SPARSELT)
    message(STATUS "[test/matmulSearch] sparseLt disabled, skipping test")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "matmulSearch|sparseLt disabled (no kernel sources for SOC=${SOC_VERSION})\n")
    return()
endif()

ops_sparse_add_gtest_tests(matmulSearch ${OPS_SPARSELT}
    NO_SRC_DIR
    EXTRA_INCLUDES
        ${CMAKE_SOURCE_DIR}/test/ltmatmul
        ${CMAKE_SOURCE_DIR}/test/ltmatmul/arch35
        ${CMAKE_SOURCE_DIR}/sparseLt/shared
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_plan
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_descriptor
        ${CMAKE_SOURCE_DIR}/sparseLt/alg_set_attribute
        ${CMAKE_SOURCE_DIR}/sparseLt/alg_get_attribute
        ${CMAKE_SOURCE_DIR}/sparseLt/prune)

# 同 target 多源组织：主文件 = TEST_P + X01~X19；xcases = X20~X23 dtype
# 超集组合；live = X31~X50 补齐 + 探针收编组。
# CSV 仍由上方宏拷贝至 target 目录（GetCasesFromCsv 相对路径不变）。
target_sources(matmulSearch_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmulSearch_test_xcases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmulSearch_test_live.cpp)

# log/log.h（经 sparseLt/shared/aclsparselt_internal.h 引入）位于 CANN
# include/op_common 下——宏默认 include 不含该路径，按 ltmatmul 先例补加。
target_include_directories(matmulSearch_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common)

# 二进制落位 build/test/matmulSearch/matmulSearch_test（harness 路径；
# ltmatmul CMake 同款手法，build.sh 运行逻辑对该路径优先匹配）。
set_target_properties(matmulSearch_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/matmulSearch)

