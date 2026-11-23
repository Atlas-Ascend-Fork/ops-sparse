# Copyright (c) 2026 AscSpMM Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT "arch22" IN_LIST SOC_ARCH_DIRS)
    message(STATUS "[cube_spmm_1d] only arch22 is implemented")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "cube_spmm_1d|Only arch22 is implemented\n")
    return()
endif()
find_package(Eigen3 3.3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/validate_catalog.py" --quiet
    RESULT_VARIABLE catalog_status)
if(NOT catalog_status EQUAL 0)
    message(FATAL_ERROR "Cube SpMM 1D catalog does not match device specializations")
endif()
ops_sparse_add_gtest_tests(cube_spmm_1d ${OPS_SPARSE} EIGEN)
if(NOT TARGET cube_spmm_1d_test)
    message(FATAL_ERROR "cube_spmm_1d delivery tests require GTest")
endif()
target_link_libraries(cube_spmm_1d_test PRIVATE Eigen3::Eigen runtime profapi)
target_compile_options(cube_spmm_1d_test PRIVATE "$<$<CONFIG:Release>:-O2>")
target_compile_definitions(cube_spmm_1d_test PRIVATE
    CUBE_SPMM_1D_CSV="${CMAKE_CURRENT_SOURCE_DIR}/arch22/cube_spmm_1d_test.csv")

target_sources(cube_spmm_1d_test PRIVATE arch22/csr_conversion_test.cpp
    ${CMAKE_SOURCE_DIR}/sparse/cube_spmm_1d/arch22/internal/csr_conversion.cpp)

# Compile the production host translation units with link-time runtime fault
# injection. This target has no device initialization and never launches an NPU.
set(operator_source "${CMAKE_SOURCE_DIR}/sparse/cube_spmm_1d/arch22")
add_executable(cube_spmm_1d_fault_test arch22/runtime_fault_test.cpp
    ${CMAKE_SOURCE_DIR}/test/frame/test_main.cpp
    ${operator_source}/cube_spmm_1d_host.cpp ${operator_source}/internal/runtime_plan.cpp
    ${operator_source}/internal/csr_conversion.cpp)
target_compile_features(cube_spmm_1d_fault_test PRIVATE cxx_std_17)
target_compile_definitions(cube_spmm_1d_fault_test PRIVATE ASCEND910B3 SPMM_1D_SUPPORTED_SOC=1)
target_include_directories(cube_spmm_1d_fault_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/sparse/common ${operator_source}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common
    ${GTEST_INCLUDE_DIRS})
target_link_directories(cube_spmm_1d_fault_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
if(TARGET GTest::gtest)
    target_link_libraries(cube_spmm_1d_fault_test PRIVATE GTest::gtest)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CUBE_SPMM_1D_GTEST REQUIRED IMPORTED_TARGET gtest)
    target_link_libraries(cube_spmm_1d_fault_test PRIVATE PkgConfig::CUBE_SPMM_1D_GTEST)
endif()
target_link_libraries(cube_spmm_1d_fault_test PRIVATE
    "-Wl,--no-as-needed" error_manager unified_dlog "-Wl,--as-needed" mmpa pthread ${CMAKE_DL_LIBS})
foreach(symbol aclrtMalloc aclrtFree aclrtMemcpy aclrtGetCurrentContext aclrtGetDevice
    aclGetDeviceCapability aclrtGetSocName aclrtMemsetAsync dladdr rtDevBinaryRegister
    rtFunctionRegister rtDevBinaryUnRegister rtGetC2cCtrlAddr rtKernelLaunch)
    target_link_options(cube_spmm_1d_fault_test PRIVATE "-Wl,--wrap=${symbol}")
endforeach()
