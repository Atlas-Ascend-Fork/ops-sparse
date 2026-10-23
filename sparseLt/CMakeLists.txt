# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# sparseLt 源文件收集：host (CXX) 与 kernel (ASC) 分离
# 全量递归收集 *.cpp，再从中剔除 _kernel.cpp 得到 host 源文件（新增模块无需修改）
file(GLOB_RECURSE SPARSELT_HOST_SRCS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE SPARSELT_KERNEL_SRCS_ALL CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp)
list(REMOVE_ITEM SPARSELT_HOST_SRCS ${SPARSELT_KERNEL_SRCS_ALL})

# 自动收集 sparseLt 子目录作为 include 路径（新增模块无需修改 CMake）
file(GLOB SPARSELT_ALL_SUBDIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*/)
set(SPARSELT_INCLUDE_DIRS "")
foreach(_dir ${SPARSELT_ALL_SUBDIRS})
    if(IS_DIRECTORY ${_dir})
        list(APPEND SPARSELT_INCLUDE_DIRS ${_dir})
    endif()
endforeach()
set(SPARSELT_INCLUDE_DIRS ${SPARSELT_INCLUDE_DIRS} PARENT_SCOPE)

set(SPARSELT_KERNEL_SRCS "")
foreach(arch_dir ${SOC_ARCH_DIRS})
    foreach(src ${SPARSELT_KERNEL_SRCS_ALL})
        if(src MATCHES "/${arch_dir}/")
            list(APPEND SPARSELT_KERNEL_SRCS ${src})
        endif()
    endforeach()
endforeach()

# ========= asc-devkit 版本检测（自包含，不依赖 cmake/ 公共文件）=========
# 读取 asc_devkit_version.h，解析 ASC_DEVKIT_MAJOR / ASC_DEVKIT_MINOR。
# sparseLt 的 prune kernel 依赖 kernel_operator.h（CANN 9.1.0+），低版本时
# 清空源文件列表使 sparseLt 库不编译，并通过 ENABLE_SPARSELT 通知 test 子目录跳过。
set(ASC_DEVKIT_MAJOR 0)
set(ASC_DEVKIT_MINOR 0)
set(ENABLE_SPARSELT TRUE)
set(_sparselt_version_header
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version/asc_devkit_version.h")

if(EXISTS "${_sparselt_version_header}")
    file(READ "${_sparselt_version_header}" _sparselt_version_content)
    if(_sparselt_version_content MATCHES "#define ASC_DEVKIT_MAJOR ([0-9]+)")
        set(ASC_DEVKIT_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_sparselt_version_content MATCHES "#define ASC_DEVKIT_MINOR ([0-9]+)")
        set(ASC_DEVKIT_MINOR "${CMAKE_MATCH_1}")
    endif()
    # arch35 的 kernel 依赖 kernel_operator.h，需 asc-devkit >= 9.1
    if("arch35" IN_LIST SOC_ARCH_DIRS AND (ASC_DEVKIT_MAJOR LESS 9 OR (ASC_DEVKIT_MAJOR EQUAL 9 AND ASC_DEVKIT_MINOR LESS 1)))
        set(ENABLE_SPARSELT FALSE)
        message(STATUS "[sparseLt] asc-devkit ${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR} < 9.1, "
                       "kernel API unavailable, skipping sparseLt library (prune)")
    endif()
else()
    set(ENABLE_SPARSELT FALSE)
    message(WARNING "[sparseLt] asc_devkit_version.h not found: ${_sparselt_version_header}, "
                    "sparseLt library will be skipped")
endif()

# No kernel sources for current SOC arch → host code references kernel launch
# functions that would be undefined at link time; disable sparseLt entirely.
if(NOT SPARSELT_KERNEL_SRCS)
    set(ENABLE_SPARSELT FALSE)
    message(STATUS "[sparseLt] No kernel sources for SOC=${SOC_VERSION} (arch=${SOC_ARCH_DIRS}), "
                   "skipping sparseLt library")
endif()

# OP_LIST filter (same idea as sparse/CMakeLists.txt: detect whether any
# requested op belongs to the sparseLt tree). Reuses ENABLE_SPARSELT: when the
# build is filtered to non-Lt ops (e.g. --ops=scatter → -DOP_LIST=scatter),
# nothing under sparseLt/ is requested, so clear sources and skip the Lt
# library. Decouples single sparse-op builds from compiling the full sparseLt
# tree.
#
# Unlike sparse/, per-name source filtering is NOT viable on the Lt tree:
# (a) test dirs are not named after source dirs (test/ltmatmul → sparseLt/
#     matmul, test/sparseLt/matmulSearch → sparseLt/matmul_search, ...);
# (b) Lt operators are interdependent (matmul_search calls aclsparseLtMatmul,
#     matmul_plan depends on matmul_descriptor, everything uses common/shared).
# Filtering by name would yield an incomplete library that fails to link.
#
# "Requested a Lt op" = ANY of:
#   1. /${op}/ matches a source path under sparseLt/ — covers source-dir-style
#      names (matmul, matmul_search, prune, prune_check, spmma_compress, ...);
#   2. <op> is an Lt TEST leaf: test/<op>/CMakeLists.txt (ltmatmul, prune, ...)
#      or test/sparseLt/<op>/CMakeLists.txt (matmulSearch, ...) exists and
#      links the Lt library (its CMakeLists.txt references OPS_SPARSELT).
#      This is REQUIRED because test leaf names never match source paths
#      (matmulSearch ≠ matmul_search, ltmatmul ≠ matmul, ...). Without it,
#      build.sh --ops=sparseLt (group dir expanded to test leaf names:
#      matmulDescriptorInit,matmulPlanInit,matmulSearch) would silently skip
#      the Lt library and keep linking the STALE build_out so — a false
#      verification hazard. New Lt tests are picked up automatically.
# Policy: if ANY op in OP_LIST is Lt-related, build the full tree (Lt operators
# are interdependent); only skip the library when nothing matches.
if(OP_LIST)
    set(_lt_requested FALSE)
    set(_lt_match_why "")
    foreach(op ${OP_LIST})
        # (1) source-path match
        foreach(src ${SPARSELT_HOST_SRCS} ${SPARSELT_KERNEL_SRCS})
            if(src MATCHES "/${op}/")
                set(_lt_requested TRUE)
                set(_lt_match_why "source path /${op}/")
                break()
            endif()
        endforeach()
        # (2) Lt test-leaf match
        if(NOT _lt_requested)
            foreach(_t_cml
                    "${CMAKE_SOURCE_DIR}/test/${op}/CMakeLists.txt"
                    "${CMAKE_SOURCE_DIR}/test/sparseLt/${op}/CMakeLists.txt")
                if(EXISTS "${_t_cml}")
                    file(READ "${_t_cml}" _t_cml_content)
                    if(_t_cml_content MATCHES "OPS_SPARSELT")
                        string(REPLACE "${CMAKE_SOURCE_DIR}/" "" _t_cml_rel "${_t_cml}")
                        set(_lt_requested TRUE)
                        set(_lt_match_why "Lt test leaf ${_t_cml_rel}")
                        break()
                    endif()
                endif()
            endforeach()
        endif()
        if(_lt_requested)
            break()
        endif()
    endforeach()

    if(_lt_requested)
        if(ENABLE_SPARSELT)
            message(STATUS "[sparseLt] OP_LIST=${OP_LIST} matched sparseLt (${_lt_match_why}); "
                           "building full sparseLt tree (Lt operators are interdependent)")
        else()
            # Anti-false-verification guard: Lt ops were explicitly requested,
            # but the library cannot be built in this environment (see the
            # devkit-version / kernel-source checks above). Warn LOUDLY instead
            # of silently skipping: a skipped build keeps the stale
            # build_out/lib64/libops_sparseLt.so in place, and anything run
            # against it would verify OLD code, not the requested one.
            message(WARNING
                "[sparseLt] ===========================================================\n"
                "[sparseLt] Lt ops requested via OP_LIST (${OP_LIST}; matched: ${_lt_match_why})\n"
                "[sparseLt] but the sparseLt library is DISABLED in this environment\n"
                "[sparseLt] (asc-devkit=${ASC_DEVKIT_MAJOR}.${ASC_DEVKIT_MINOR}, SOC=${SOC_VERSION},\n"
                "[sparseLt] arch=${SOC_ARCH_DIRS}). The library is SKIPPED: build_out keeps\n"
                "[sparseLt] the PREVIOUS libops_sparseLt.so and all Lt tests will be skipped.\n"
                "[sparseLt] Results run against the stale library are NOT a verification.\n"
                "[sparseLt] ===========================================================")
        endif()
    else()
        set(SPARSELT_HOST_SRCS "")
        set(SPARSELT_KERNEL_SRCS "")
        set(ENABLE_SPARSELT FALSE)
        message(STATUS "[sparseLt] OP_LIST=${OP_LIST} matches no path under sparseLt/, "
                       "ENABLE_SPARSELT=FALSE")
    endif()
endif()

if(NOT ENABLE_SPARSELT)
    # 清空源文件列表，使根 CMakeLists.txt 的 if(ALL_SPARSELT_SRC_FILES) 跳过库构建
    set(SPARSELT_HOST_SRCS "")
    set(SPARSELT_KERNEL_SRCS "")
endif()

message(STATUS "[sparseLt] ASC_DEVKIT_MAJOR=${ASC_DEVKIT_MAJOR}, ASC_DEVKIT_MINOR=${ASC_DEVKIT_MINOR}, "
               "ENABLE_SPARSELT=${ENABLE_SPARSELT}")

set(ALL_SPARSELT_SRC_FILES ${SPARSELT_HOST_SRCS} ${SPARSELT_KERNEL_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_HOST_SRC_FILES ${SPARSELT_HOST_SRCS} PARENT_SCOPE)
set(ALL_SPARSELT_KERNEL_SRC_FILES ${SPARSELT_KERNEL_SRCS} PARENT_SCOPE)
set(ENABLE_SPARSELT ${ENABLE_SPARSELT} PARENT_SCOPE)

# tensor_api (CANN-provided at asc/include/tensor_api) pulls in implementation
# headers from asc/impl/ and asc/impl/c_api/. In NPU build mode the ASC cmake
# module does NOT add asc/impl/c_api to the compiler system include path (it
# only does so in --run-mode=cpu via CMakeASCInformation.cmake). Without this
# path the deep include chain (tensor_api -> impl/tensor_api -> impl/c_api ->
# instr_impl/...) fails with "file not found".
#
# The ops_sparseLt target is created in the parent (root) CMakeLists.txt AFTER
# add_subdirectory(sparseLt) returns, so target_include_directories cannot be
# called here directly. Use cmake_language(DEFER DIRECTORY) to schedule the call
# at the end of the root directory processing, by which time the target exists.
# This uses the CANN-provided tensor_api (no ops-tensor dependency).
#
# DEFER scope pitfall (verified experimentally): CALL arguments are evaluated
# when the deferred call RUNS (root-directory scope), not when it is recorded
# here. Directory-local variables such as _tensor_api_base are out of scope
# there and expand to EMPTY — this silently turned the include flags into
# "-I/include -I" and broke every tensor_api include ("tensor_api/tensor.h
# file not found"). Only root-scope-visible variables (ASCEND_CANN_PACKAGE_PATH,
# CMAKE_SYSTEM_PROCESSOR, OPS_SPARSELT) may be referenced in the CALL arguments;
# spell out the tensor_api paths inline instead of using a local variable.
if(ENABLE_SPARSELT AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR}
        CALL target_include_directories ${OPS_SPARSELT} PRIVATE
            "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/c_api"
            "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/tensor_api"
            "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/tensor_api/include"
        )
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR}
        CALL target_compile_options ${OPS_SPARSELT} PRIVATE
            $<$<COMPILE_LANGUAGE:ASC>:-I${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/tensor_api/include>
            $<$<COMPILE_LANGUAGE:ASC>:-I${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/tensor_api>
        )
endif()
