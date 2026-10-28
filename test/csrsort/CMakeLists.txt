# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set(_op_name csrsort)
set(_target ${_op_name}_test)
set(_test_src "")
set(_src_arch "")

foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/${_target}.cpp")
    if(EXISTS ${_candidate})
        set(_test_src ${_candidate})
        set(_src_arch ${arch_dir})
        break()
    endif()
endforeach()

if(NOT _test_src)
    message(STATUS "[test/${_op_name}] no test sources for SOC=${SOC_VERSION} (SOC_ARCH_DIRS=${SOC_ARCH_DIRS}), skipping ${_target}")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "${_op_name}|no test sources for SOC=${SOC_VERSION} (arch=${SOC_ARCH_DIRS})\n")
    return()
endif()

if(NOT IS_DIRECTORY "${CMAKE_SOURCE_DIR}/sparse/${_op_name}/${_src_arch}")
    message(FATAL_ERROR "[test/${_op_name}] test arch '${_src_arch}' has no matching source at sparse/${_op_name}/${_src_arch}")
endif()

add_executable(${_target} ${_test_src})
target_compile_features(${_target} PRIVATE cxx_std_17)

target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparse/${_op_name}/${_src_arch}
    ${CMAKE_SOURCE_DIR}/sparse/common
    ${CMAKE_SOURCE_DIR}/test/frame
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include/adv_api
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(${_target} PRIVATE
    ${OPS_SPARSE}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/lib64/libtiling_api.a
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
)

find_package(GTest REQUIRED)
target_link_libraries(${_target} PRIVATE GTest::gtest)
