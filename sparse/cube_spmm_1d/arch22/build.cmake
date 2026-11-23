# Copyright (c) 2026 AscSpMM Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(local "${CMAKE_CURRENT_LIST_DIR}")
set(host_sources "${local}/cube_spmm_1d_host.cpp"
    "${local}/internal/runtime_plan.cpp" "${local}/internal/csr_conversion.cpp")
set_source_files_properties(${host_sources} PROPERTIES
    LANGUAGE CXX COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")
set_source_files_properties("${local}/internal/runtime_plan.cpp"
    PROPERTIES COMPILE_DEFINITIONS "ASCEND910B3;SPMM_1D_SUPPORTED_SOC=1")
target_compile_features(${target} PRIVATE cxx_std_17)
target_include_directories(${target} PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
target_link_directories(${target} PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_libraries(${target} PRIVATE
    "-Wl,--start-group"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendc_runtime.a"
    runtime ascendcl profapi unified_dlog mmpa pthread
    "-Wl,--end-group")
set(output "${CMAKE_BINARY_DIR}/cube_spmm_1d_kernels")
file(GLOB dependencies "${local}/*.h" "${local}/internal/*.h" "${local}/internal/*.inc")
set(includes "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/interface"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/impl" "${local}")
set(names workspace_unified direct direct_pipeline fusion2 fusion4 fusion8 nmajor)
foreach(family IN LISTS names)
    set(source "${local}/spmm_1d_kernel.cce")
    set(definitions ASCEND910B3 "SPMM_KERNEL_NAME=spmm_1d_kernel_${family}")
    if(family MATCHES "^direct")
        list(APPEND definitions SPMM_KERNEL_FAMILY_DIRECT=1)
        if(family STREQUAL "direct_pipeline")
            list(APPEND definitions SPMM_DIRECT_EXTENSIONS=1)
        endif()
    else()
        list(APPEND definitions SPMM_KERNEL_FAMILY_WORKSPACE=1)
        if(family MATCHES "^fusion([248])$")
            set(source "${local}/spmm_1d_fusion.cce")
            list(APPEND definitions "SPMM_FUSION_GROUP=${CMAKE_MATCH_1}")
        elseif(family STREQUAL "nmajor")
            list(APPEND definitions SPMM_NMAJOR_EXTENSIONS=1)
        endif()
    endif()
    ops_sparse_add_cce_mixed_family(${target} "${source}" "spmm_1d_kernel_${family}"
        OUTPUT_DIR "${output}" DEFINITIONS ${definitions}
        INCLUDES ${includes} DEPENDS ${dependencies})
    install(FILES "${OPS_SPARSE_LAST_CCE_BINARY}" DESTINATION lib64/cube_spmm_1d_kernels)
endforeach()
