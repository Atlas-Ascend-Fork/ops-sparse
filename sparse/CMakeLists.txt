# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# ============================================================================
# 架构特定源文件收集逻辑
# ============================================================================
# 原理：
#   1. 首先收集所有 .cpp 源文件作为基础源文件
#   2. 根据 ARCH_SPECIFIC_DIRS 过滤掉所有架构特定目录中的文件
#   3. 根据 SOC_ARCH_DIRS 重新添加当前 SOC 对应的架构特定源文件
#
# 说明：
#   file(GLOB_RECURSE ... **/*.cpp) 模式无法正确匹配嵌套目录，
#   故使用路径字符串匹配来正确识别架构特定文件
# ============================================================================

# 收集基础源文件（所有 .cpp 文件）
file(GLOB_RECURSE BASE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 过滤构建树残留：算子目录内独立构建的 build/ 产物不得进入源清单
#（残留源会污染仓顶构建的 GLOB 收集；根构建目录在本目录之外不受影响）
list(FILTER BASE_SRC_FILES EXCLUDE REGEX "/build/")
message(STATUS "[sparse] Total source files before filtering: ${BASE_SRC_FILES}")

# -------------------------------------------------------------------------------------------------
# 步骤1：过滤掉所有架构特定目录中的文件
#   - ARCH_SPECIFIC_DIRS 在根 CMakeLists.txt 中定义（如 arch35, arch22, arch20）
#   - 使用路径匹配替代 GLOB 模式，确保正确识别嵌套目录中的架构特定文件
# -------------------------------------------------------------------------------------------------
foreach(arch_dir ${ARCH_SPECIFIC_DIRS})
  set(arch_dir_srcs "")
  foreach(src ${BASE_SRC_FILES})
    # 使用字符串匹配检查文件路径是否包含架构目录
    if(src MATCHES "/${arch_dir}/")
      list(APPEND arch_dir_srcs ${src})
    endif()
  endforeach()
  
  if(arch_dir_srcs)
    message(STATUS "[sparse] Filtering out ${arch_dir} sources: ${arch_dir_srcs}")
    foreach(arch_src ${arch_dir_srcs})
      list(REMOVE_ITEM BASE_SRC_FILES ${arch_src})
    endforeach()
  endif()
endforeach()

# torch_extension/csrc 由 PyTorch JIT 单独编译，不能进入 libops_sparse。
set(torch_extension_srcs "")
foreach(src ${BASE_SRC_FILES})
  if(src MATCHES "/torch_extension/")
    list(APPEND torch_extension_srcs ${src})
  endif()
endforeach()
foreach(torch_extension_src ${torch_extension_srcs})
  list(REMOVE_ITEM BASE_SRC_FILES ${torch_extension_src})
endforeach()

# -------------------------------------------------------------------------------------------------
# 步骤2：收集当前 SOC 对应的架构特定源文件
#   - SOC_ARCH_DIRS 根据 SOC_VERSION 在根 CMakeLists.txt 中通过 get_soc_arch_dirs() 确定
#   - 例如：ascend910b -> arch22, ascend950 -> arch35, ascend310p -> arch20
# -------------------------------------------------------------------------------------------------
set(ARCH_SRC_FILES "")
if(SOC_ARCH_DIRS)
  message(STATUS "[sparse] SOC_ARCH_DIRS specified, collecting arch-specific sources: ${SOC_ARCH_DIRS}")
  foreach(arch_dir ${SOC_ARCH_DIRS})
    # 收集所有源文件，然后筛选出匹配当前架构目录的文件
    file(GLOB_RECURSE all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    list(FILTER all_srcs EXCLUDE REGEX "/build/")
    set(arch_dir_srcs "")
    foreach(src ${all_srcs})
      if(src MATCHES "/${arch_dir}/")
        list(APPEND arch_dir_srcs ${src})
      endif()
    endforeach()
    
    if(arch_dir_srcs)
      message(STATUS "[sparse] Adding ${arch_dir} sources: ${arch_dir_srcs}")
      list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
    else()
      message(WARNING "[sparse] Arch directory '${arch_dir}' specified but no source files found")
    endif()
  endforeach()
else()
  message(STATUS "[sparse] No SOC_ARCH_DIRS specified, using generic implementation only")
endif()

# -------------------------------------------------------------------------------------------------
# 步骤3：合并源文件列表并输出到父作用域
# -------------------------------------------------------------------------------------------------
set(ALL_SPARSE_SRC_FILES "")
list(APPEND ALL_SPARSE_SRC_FILES ${BASE_SRC_FILES} ${ARCH_SRC_FILES})

# 去重（防止同一文件被多次添加）
if(ALL_SPARSE_SRC_FILES)
  list(REMOVE_DUPLICATES ALL_SPARSE_SRC_FILES)
endif()

# OP_LIST 过滤：仅编译指定算子的源文件（common/ 始终保留）
if(OP_LIST)
  message(STATUS "[sparse] OP_LIST filter active: ${OP_LIST}")
  set(_filtered_srcs "")
  foreach(src ${ALL_SPARSE_SRC_FILES})
    if(src MATCHES "/common/")
      list(APPEND _filtered_srcs ${src})
      continue()
    endif()
    foreach(op ${OP_LIST})
      if(src MATCHES "/${op}/")
        list(APPEND _filtered_srcs ${src})
        break()
      endif()
    endforeach()
  endforeach()
  set(ALL_SPARSE_SRC_FILES ${_filtered_srcs})
  list(LENGTH _filtered_srcs _filtered_count)
  message(STATUS "[sparse] Filtered to ${OP_LIST}: ${_filtered_count} files")
endif()

# 输出统计信息
list(LENGTH BASE_SRC_FILES base_count)
list(LENGTH ARCH_SRC_FILES arch_count)
list(LENGTH ALL_SPARSE_SRC_FILES total_count)
message(STATUS "[sparse] Source file summary: ${base_count} generic + ${arch_count} arch-specific = ${total_count} total")

# 设置父作用域变量
set(ALL_SPARSE_SRC_FILES ${ALL_SPARSE_SRC_FILES} PARENT_SCOPE)
