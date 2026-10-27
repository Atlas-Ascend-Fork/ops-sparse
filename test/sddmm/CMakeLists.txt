# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseSDDMM GTest + CSV parameterized tests.
# Older GTest releases do not invoke SetUpTestSuite, leaving the shared ACL
# fixture uninitialized even though the test executable links successfully.
find_package(GTest 1.10 REQUIRED)
ops_sparse_add_gtest_tests(sddmm ${OPS_SPARSE} WARN_ON_MISSING_SRC)
if(TARGET sddmm_test)
    target_compile_features(sddmm_test PRIVATE cxx_std_17)
    if("arch22" IN_LIST SOC_ARCH_DIRS)
        target_sources(sddmm_test PRIVATE arch22/sddmm_contract_test.cpp)
    endif()
    get_target_property(_sddmm_test_sources sddmm_test SOURCES)
    set(_sddmm_arch35_test "${CMAKE_CURRENT_SOURCE_DIR}/arch35/sddmm_test.cpp")
    if(_sddmm_arch35_test IN_LIST _sddmm_test_sources)
        target_sources(sddmm_test PRIVATE
            arch35/sddmm_test_bsr_support.cpp
            arch35/sddmm_test_csr_support.cpp
            arch35/sddmm_test_complex_support.cpp
            arch35/sddmm_test_pattern.cpp
            arch35/sddmm_test_workspace_support.cpp)
        if(NOT OP_LIST OR "spmm" IN_LIST OP_LIST)
            target_compile_definitions(sddmm_test PRIVATE SDDMM_TEST_WITH_SPMM=1)
        endif()
    endif()
endif()

# Standalone Event-timed benchmark. Keep it beside the GTest binary so both
# targets use the same freshly built operator library and arch35 headers.
set(_sddmm_perf_src "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    set(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${arch_dir}/sddmm_perf.cpp")
    if(EXISTS ${_candidate})
        set(_sddmm_perf_src ${_candidate})
        set(_sddmm_perf_arch ${arch_dir})
        break()
    endif()
endforeach()

if(_sddmm_perf_src)
    add_executable(sddmm_perf ${_sddmm_perf_src})
    target_compile_features(sddmm_perf PRIVATE cxx_std_17)
    set_target_properties(sddmm_perf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/sddmm)
    target_include_directories(sddmm_perf PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/sparse/common
        ${CMAKE_SOURCE_DIR}/sparse/sddmm/${_sddmm_perf_arch}
        ${CMAKE_SOURCE_DIR}/test/frame
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${_sddmm_perf_arch}
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    target_link_directories(sddmm_perf PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
    target_link_libraries(sddmm_perf PRIVATE
        "-Wl,--start-group"
        ${OPS_SPARSE}
        ${_ops_sparse_ascendc_runtime_lib}
        ${_ops_sparse_ascendcl_lib}
        ${_ops_sparse_libc_sec}
        ascendcl
        tiling_api
        platform
        unified_dlog
        mmpa
        "-Wl,--end-group"
        pthread
    )
endif()
