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
# GTest + CSV parameterized test for aclsparseLtSpMMAPruneCheck (2:4 校验).
#
# This test exercises the check stage of the aclsparseLt chain:
#   aclsparseLtInit
#   -> StructuredDescriptorInit(稀疏侧) / DenseDescriptorInit(其余)
#   -> MatmulDescriptorInit
#   -> SpMMAPruneCheck     (d_in -> d_valid)     [core API under test]
#
# 无 AlgSelection / AlgSetAttribute / Plan 环节（对齐需求 §2.2 schema）。
# d_valid（Device int 标量）拷回 host 后与 CPU golden 精确相等（bit-exact，
# 零容差），返回码与预期枚举精确相等（2.1 §2.3 口径表）。
#
# 复用（2.1 §8.1 框架强制规则）：test/frame/ 公共头（csv_loader/sparse_test/
# descriptor_manager/verify/test_main.cpp）、prune 家族 wrapper 既有件
# （EventGuard/NpuDtypeTrait/PreparePhysicalA：test/prune/arch35/prune_npu_wrapper.h；
# SparseLtHandleGuard/MatDescrGuard/MatmulDescrGuard：
# test/sparseLt/matmulDescriptorInit/matmulDescriptorInit_npu_wrapper.h）。
# =============================================================================

# ---- 版本拦截：sparseLt 库依赖 kernel API（CANN 9.1.0+），低版本跳过 ----
if(NOT ENABLE_SPARSELT)
    message(STATUS "[test/prune_check] ENABLE_SPARSELT=FALSE (requires asc-devkit >= 9.1), skipping prune_check_test")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "prune_check|requires asc-devkit >= 9.1 (ENABLE_SPARSELT=FALSE)\n")
    return()
endif()

set(_target prune_check_test)
set(_src_arch "")
set(_test_src "")

foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/prune_check_test.cpp")
    if(EXISTS ${_candidate})
        set(_test_src ${_candidate})
        set(_src_arch ${arch_dir})
        break()
    endif()
endforeach()

if(NOT _test_src)
    message(STATUS "[test/prune_check] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${_target}")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "prune_check|no test sources for SOC=${SOC_VERSION} (arch=${SOC_ARCH_DIRS})\n")
    return()
endif()

# ---- Locate GTest (main() is provided by test/frame/test_main.cpp) ----
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST QUIET gtest)
    endif()
    if(NOT GTEST_FOUND)
        message(WARNING "[test/prune_check] GTest not found, skipping ${_target}")
        file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
            "prune_check|GTest not found\n")
        return()
    endif()
else()
    set(GTEST_LIBRARIES GTest::gtest)
    set(GTEST_INCLUDE_DIRS "")
endif()

add_executable(${_target} ${_test_src} ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp)
target_compile_features(${_target} PRIVATE cxx_std_17)

# Pin binary output to build/test/prune_check/ (build.sh expectation).
set_target_properties(${_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/prune_check)

target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/sparseLt/prune_check/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparseLt/shared
    ${CMAKE_SOURCE_DIR}/sparseLt/matmul_descriptor
    ${CMAKE_SOURCE_DIR}/test/frame
    ${CMAKE_SOURCE_DIR}/test/sparseLt/matmulDescriptorInit
    ${CMAKE_SOURCE_DIR}/test/prune                # prune_test_util.h / prune_golden.h
    ${CMAKE_SOURCE_DIR}/test/prune/${_src_arch}   # prune_npu_wrapper.h（EventGuard/NpuDtypeTrait/PreparePhysicalA 复用）
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}
    $ENV{LINUX_INCLUDE_PATH}
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(${_target} PRIVATE
    ${OPS_SPARSELT}
    ${_ops_sparse_ascendcl_lib}
    ${GTEST_LIBRARIES}
    pthread
)

# ---- Copy CSV next to the binary so the test loads it by relative path ----
set(_csv_file "${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}/prune_check_test.csv")
if(EXISTS ${_csv_file} AND TARGET ${_target})
    add_custom_command(TARGET ${_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${_csv_file} $<TARGET_FILE_DIR:${_target}>/prune_check_test.csv
        COMMENT "Copying prune_check_test.csv to build dir"
    )
endif()
