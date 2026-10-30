# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# GTest + CSV parameterized test for the full aclsparseLt chain:
#   aclsparseLtMatmul (with built-in setAttr → getAttr → plan → prune → matmul)
#
# The matmul path explicitly asserts getAttr roundtrip, so standalone
# alg_set_attribute / alg_get_attribute CSV suites have been removed.
# AlgSetAttributeExceptionTest (TEST_F) is retained for parameter validation.
#
# Source layout (this directory = test/ltmatmul/):
#   arch35/ltmatmul_test.cpp          (matmul CSV + TEST_F + alg_set exception)
#   arch35/ltmatmul_test_neg.cpp      (WB_L2_42~47 白盒负向系，主文件 NBNC 拆分)
#   arch35/ltmatmul_test.csv          (matmul CSV cases)
#   arch35/ltmatmul_perf_test.cpp     (perf benchmark)
#   ltmatmul_golden.h / ltmatmul_golden_types.h / ltmatmul_golden_int8.h
#   ltmatmul_param.h (MatmulParam only)
#   arch35/ltmatmul_npu_wrapper.h (all NPU wrappers merged)
#   arch35/ltmatmul_exception_helper.h
# =============================================================================

if(NOT ENABLE_SPARSELT)
    message(STATUS "[test/ltmatmul] ENABLE_SPARSELT=FALSE (requires asc-devkit >= 9.1), skipping ltmatmul_test")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "ltmatmul|requires asc-devkit >= 9.1 (ENABLE_SPARSELT=FALSE)\n")
    return()
endif()

set(_target ltmatmul_test)
set(_src_arch "")
set(_test_src "")

foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/ltmatmul_test.cpp")
    if(EXISTS ${_candidate})
        set(_test_src ${_candidate})
        set(_src_arch ${arch_dir})
        break()
    endif()
endforeach()

if(NOT _test_src)
    message(STATUS "[test/ltmatmul] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${_target}")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "ltmatmul|no test sources for SOC=${SOC_VERSION} (arch=${SOC_ARCH_DIRS})\n")
    return()
endif()

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST QUIET gtest)
    endif()
    if(NOT GTEST_FOUND)
        message(WARNING "[test/ltmatmul] GTest not found, skipping ${_target}")
        file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
            "ltmatmul|GTest not found\n")
        return()
    endif()
else()
    set(GTEST_LIBRARIES GTest::gtest)
    set(GTEST_INCLUDE_DIRS "")
endif()

add_executable(${_target} ${_test_src} ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)
target_compile_features(${_target} PRIVATE cxx_std_17)

# WB_L2 负向系（codecheck7 拆分：主文件 NBNC 超 2000）——同 target 多源。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_neg.cpp")
    target_sources(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_neg.cpp)
endif()

# D2 BATCH_STRIDE=0 跨批广播手写用例族（BC1-BC8 + INT8 代表组）——同 target 多源。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_broadcast.cpp")
    target_sources(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_broadcast.cpp)
endif()

# D5 16B 对齐回归锚点族（三处 DataCopyPad 源 32B->16B 放宽的回归锚点）。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_align16.cpp")
    target_sources(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test_align16.cpp)
endif()

# B1 白盒 forged 系（FastPath 混合态回归——D5 决策）——同 target 多源。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_whitebox_b1.cpp")
    target_sources(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_whitebox_b1.cpp)
endif()

# PR 检视意见① 白盒回归（fill_splitk_tiling_dims int64 域化——host 纯函数
# 锻造大 k，断言 kSegLen/validSplitK 不回绕 + k==0 除零防御）——同 target 多源。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_whitebox_splitk.cpp")
    target_sources(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_whitebox_splitk.cpp)
endif()

set_target_properties(${_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/ltmatmul)

target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/sparseLt/matmul/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/alg_set_attribute/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/alg_get_attribute/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/prune/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/shared
    ${CMAKE_SOURCE_DIR}/sparseLt/common
    ${CMAKE_SOURCE_DIR}/sparseLt/matmul_descriptor
    ${CMAKE_SOURCE_DIR}/sparseLt/matmul_plan
    ${CMAKE_SOURCE_DIR}/test/frame
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    $ENV{LINUX_INCLUDE_PATH}
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(${_target} PRIVATE
    ${OPS_SPARSELT}
    ${_ops_sparse_ascendcl_lib}
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    ${GTEST_LIBRARIES}
    pthread
)

# ---- Copy CSV file next to the binary so tests load it by relative path ----
set(_csv_ltmatmul "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_test.csv")
if(EXISTS ${_csv_ltmatmul} AND TARGET ${_target})
    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${_csv_ltmatmul} $<TARGET_FILE_DIR:${_target}>/ltmatmul_test.csv
        COMMENT "Copying ltmatmul_test.csv to build dir"
    )
endif()

# =============================================================================
# Performance baseline benchmark (standalone, no GTest).
# Reuses npu_wrapper EventGuard timing; warmup + measured iterations.
# =============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_perf_test.cpp")
    set(_perf_target ltmatmul_perf_test)
    add_executable(${_perf_target}
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/ltmatmul_perf_test.cpp)
    target_compile_features(${_perf_target} PRIVATE cxx_std_17)
    set_target_properties(${_perf_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/ltmatmul)
    target_include_directories(${_perf_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul/${_src_arch}
        ${CMAKE_SOURCE_DIR}/sparseLt/alg_set_attribute/${_src_arch}
        ${CMAKE_SOURCE_DIR}/sparseLt/alg_get_attribute/${_src_arch}
        ${CMAKE_SOURCE_DIR}/sparseLt/prune/${_src_arch}
        ${CMAKE_SOURCE_DIR}/sparseLt/shared
        ${CMAKE_SOURCE_DIR}/sparseLt/common
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_descriptor
        ${CMAKE_SOURCE_DIR}/sparseLt/matmul_plan
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    target_link_libraries(${_perf_target} PRIVATE
        ${OPS_SPARSELT}
        ${_ops_sparse_ascendcl_lib}
        $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
        pthread
    )
endif()
