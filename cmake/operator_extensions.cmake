# Copyright (c) 2026 AscSpMM Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Optional operator-local build declarations. Selection matches OP_LIST/SOC.
# This hook is shared by backends that need generated or external device objects.
function(ops_sparse_configure_extensions target)
    file(GLOB operators LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/sparse/*")
    foreach(directory IN LISTS operators)
        get_filename_component(operator "${directory}" NAME)
        if(OP_LIST AND NOT operator IN_LIST OP_LIST)
            continue()
        endif()
        foreach(arch IN LISTS SOC_ARCH_DIRS)
            set(extension "${directory}/${arch}/build.cmake")
            if(EXISTS "${extension}")
                include("${extension}")
            endif()
        endforeach()
    endforeach()
endfunction()

function(ops_sparse_find_cce_tools compiler linker)
    get_filename_component(asc_bin "${CMAKE_ASC_COMPILER}" DIRECTORY)
    get_filename_component(lld_bin "${CMAKE_ASC_LLD_LINKER}" DIRECTORY)
    set(tool_hints "${asc_bin}" "${lld_bin}"
        "${ASCEND_CANN_PACKAGE_PATH}/bin"
        "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ccec_compiler/bin"
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler/bin"
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/bin")
    # Keep the legacy ccec driver when supplied; newer packages expose bisheng.
    find_program(OPS_SPARSE_CCEC NAMES ccec HINTS ${tool_hints} NO_DEFAULT_PATH)
    if(NOT OPS_SPARSE_CCEC AND EXISTS "${CMAKE_ASC_COMPILER}")
        set(OPS_SPARSE_CCEC "${CMAKE_ASC_COMPILER}")
    endif()
    if(EXISTS "${CMAKE_ASC_LLD_LINKER}")
        set(OPS_SPARSE_LLD "${CMAKE_ASC_LLD_LINKER}")
    else()
        find_program(OPS_SPARSE_LLD NAMES ld.lld HINTS ${tool_hints} NO_DEFAULT_PATH)
    endif()
    if(NOT OPS_SPARSE_CCEC OR NOT OPS_SPARSE_LLD)
        message(FATAL_ERROR "CCE mixed kernels require a CANN ccec/bisheng driver and ld.lld")
    endif()
    set(${compiler} "${OPS_SPARSE_CCEC}" PARENT_SCOPE)
    set(${linker} "${OPS_SPARSE_LLD}" PARENT_SCOPE)
endfunction()

function(ops_sparse_add_cce_mixed_family target source name)
    cmake_parse_arguments(ARG "" "OUTPUT_DIR" "DEFINITIONS;INCLUDES;DEPENDS" ${ARGN})
    ops_sparse_find_cce_tools(cce_compiler cce_linker)
    set(options -std=c++17 -c -x cce --cce-aicore-only
        "$<$<CONFIG:Debug>:-O0>" "$<$<NOT:$<CONFIG:Debug>>:-O2>")
    foreach(definition IN LISTS ARG_DEFINITIONS)
        list(APPEND options "-D${definition}")
    endforeach()
    foreach(directory IN LISTS ARG_INCLUDES)
        list(APPEND options "-I${directory}")
    endforeach()
    file(MAKE_DIRECTORY "${ARG_OUTPUT_DIR}")
    foreach(side aic aiv)
        if(side STREQUAL "aic")
            set(architecture dav-c220-cube)
        else()
            set(architecture dav-c220-vec)
        endif()
        add_custom_command(OUTPUT "${ARG_OUTPUT_DIR}/${name}_${side}.o"
            COMMAND ${cce_compiler} ${options} "--cce-aicore-arch=${architecture}"
                "${source}" -o "${ARG_OUTPUT_DIR}/${name}_${side}.o"
            DEPENDS "${source}" ${ARG_DEPENDS} VERBATIM COMMAND_EXPAND_LISTS)
    endforeach()
    set(binary "${ARG_OUTPUT_DIR}/${name}.o")
    add_custom_command(OUTPUT "${binary}"
        COMMAND ${cce_linker} -m aicorelinux -Ttext=0
            "${ARG_OUTPUT_DIR}/${name}_aic.o" "${ARG_OUTPUT_DIR}/${name}_aiv.o"
            --allow-multiple-definition -static -o "${binary}"
        DEPENDS "${ARG_OUTPUT_DIR}/${name}_aic.o" "${ARG_OUTPUT_DIR}/${name}_aiv.o" VERBATIM)
    add_custom_target(${name}_binary DEPENDS "${binary}")
    add_dependencies(${target} ${name}_binary)
    set(OPS_SPARSE_LAST_CCE_BINARY "${binary}" PARENT_SCOPE)
endfunction()
