# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_sparse VERSION 1.0.0)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b3)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3510）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-2002")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 版本返回需要编译的架构目录列表
#
# 支持的映射关系：
#   - ascend910b*    -> arch22 (DAV-2201 架构)
#   - ascend910_93*  -> arch22 (DAV-2201 架构)
#   - ascend950*     -> arch35 (DAV-3510 架构)
#   - ascend310p*    -> arch20 (DAV-2002 架构)
#
# 函数参数：
#   soc_version: 输入的 SOC 版本字符串（如 ascend910b3, ascend950）
#   arch_dirs:   输出变量名，返回架构目录列表
# ============================================================================
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  if(soc_lower MATCHES "^ascend910b")
    list(APPEND dirs "arch22")
    message(STATUS "  -> SOC [${soc_version}] matched arch22 (DAV-2201)")
  elseif(soc_lower MATCHES "^ascend910_93")
    list(APPEND dirs "arch22")
    message(STATUS "  -> SOC [${soc_version}] matched arch22 (DAV-2201)")
  elseif(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
    message(STATUS "  -> SOC [${soc_version}] matched arch35 (DAV-3510)")
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
    message(STATUS "  -> SOC [${soc_version}] matched arch20 (DAV-2002)")
  endif()

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 架构特定目录列表（全局定义）
# 这些目录包含架构特定的优化实现，会从通用源文件中排除，再根据 SOC 选择性添加
set(ARCH_SPECIFIC_DIRS arch35 arch22 arch20 CACHE INTERNAL "Architecture specific directories")

# CANN installations expose either ASCEND_HOME_PATH or the newer
# ASCEND_TOOLKIT_HOME. Prefer the former for compatibility with the existing
# build scripts, but accept the latter so A5 builds do not depend on a shell
# alias being exported.
if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
else()
  set(ASCEND_CANN_PACKAGE_PATH "")
endif()
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
# Default Release when unspecified; FORCE only here so an empty cache entry is corrected,
# while -DCMAKE_BUILD_TYPE=Debug from the command line is still honored.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
endif()
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/build_out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)
set(OP_LIST "" CACHE STRING "Operator list to build (semicolon-separated, e.g. 'spmv;spmm')")

# ========= 构建动态库 =========
find_package(ASC REQUIRED)

# sparseLt library (built first, like blasLt in ops-blas)
set(ALL_SPARSELT_SRC_FILES "")
set(OPS_SPARSELT ops_sparseLt)

add_subdirectory(sparseLt)

if(ALL_SPARSELT_SRC_FILES)
    project(${OPS_SPARSELT} LANGUAGES ASC CXX)
    add_library(${OPS_SPARSELT} SHARED ${ALL_SPARSELT_SRC_FILES})
    set_source_files_properties(
        ${ALL_SPARSELT_KERNEL_SRC_FILES}
        PROPERTIES LANGUAGE ASC
    )
    set_source_files_properties(
        ${ALL_SPARSELT_HOST_SRC_FILES}
        PROPERTIES LANGUAGE CXX
    )
    target_include_directories(${OPS_SPARSELT} PRIVATE
        ./include
        ./sparseLt
        ${SPARSELT_INCLUDE_DIRS}
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/utils
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
    )
    target_compile_options(${OPS_SPARSELT} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    )
    target_link_directories(${OPS_SPARSELT} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
    target_link_libraries(${OPS_SPARSELT} PRIVATE
        tiling_api
        platform
        ascendcl
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendc_runtime.a
        unified_dlog
        mmpa
        profapi
    )
endif()

# sparse operators
set(ALL_SPARSE_SRC_FILES "")
set(OPS_SPARSE ops_sparse)

add_subdirectory(sparse)

project(${OPS_SPARSE} LANGUAGES ASC CXX)

add_library(${OPS_SPARSE} SHARED ${ALL_SPARSE_SRC_FILES})

set_source_files_properties(
    ${ALL_SPARSE_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

# 仓内头文件必须排在 CANN 安装目录之前。已安装的 ops-sparse 运行包会在 CANN 目录下
# 留一份同名的 cann_ops_sparse.h；使用旧头文件会导致本仓新增枚举无法解析。

# The A5 SpMV kernel uses the SIMD+SIMT hybrid model: an ordinary AIV entry
# coordinates __simt_vf__ work through asc_vf_call.  Do not pass
# --enable-simt here; that option selects the pure-SIMT compilation mode and
# rejects kernel_operator.h / other SIMD headers required by the hybrid TU.
target_include_directories(${OPS_SPARSE} BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse/common
)

target_include_directories(${OPS_SPARSE} PRIVATE
    ./include
    ./sparse/common
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
)

target_compile_options(${OPS_SPARSE} PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:SHELL:-iquote ${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<$<COMPILE_LANGUAGE:ASC>:SHELL:-iquote ${CMAKE_CURRENT_SOURCE_DIR}/sparse/common>"
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)

target_link_libraries(${OPS_SPARSE} PRIVATE
    tiling_api
    platform
    graph  # ge::Shape / ge::DataType, used by coosort host GetSortMaxMinTmpSize
    register
)

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_SPARSE}
    LIBRARY DESTINATION lib64
)
if(TARGET ${OPS_SPARSELT})
    install(TARGETS ${OPS_SPARSELT}
        LIBRARY DESTINATION lib64
    )
endif()
install(FILES
    include/cann_ops_sparse.h
    include/cann_ops_sparseLt.h
    DESTINATION include
)

# ========= 构建测试程序 =========
include(cmake/test.cmake)
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

# Optional PyTorch/torch_npu bridge.  Keep this opt-in because the core
# aclSPARSE library must remain buildable in environments without PyTorch.
option(BUILD_TORCH_ADAPTER "Build the PyTorch/torch_npu SpMV bridge" OFF)
if(BUILD_TORCH_ADAPTER)
    add_subdirectory(torch_npu)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    add_dependencies(${OPS_SPARSE} gen_ops_sparse_version_info)
    pack(${SOC_VERSION})
endif()
