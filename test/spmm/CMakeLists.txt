# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# SPMM V1 仅在 Ascend 950PR（SIMT 扩展 + dav-3510）上交付，与 src/spmm/ 的 SOC 网关保持一致。
# SPMM arch22 在 dav-2201 上交付，覆盖 Ascend910_93 系列与 Ascend910B 系列（二者均映射到 arch22，
# 见根 CMakeLists 的 SOC_VERSION -> NPU_ARCH 映射），故统一编译 arch22 测试。
if(SOC_VERSION_LOWER MATCHES "^ascend950")
    add_executable(spmm_test
        arch35/spmm_test.cpp
    )
    target_include_directories(spmm_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    target_link_libraries(spmm_test PRIVATE
        ${OPS_SPARSE}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
    )
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93" OR SOC_VERSION_LOWER MATCHES "^ascend910b")
    add_executable(spmm_test
        arch22/spmm_test.cpp
        arch22/spmm_defect_extra_test.cpp
        arch22/spmm_precision_test.cpp
    )
    target_include_directories(spmm_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    # libc_sec 是 memcpy_s/snprintf_s 的实现，ops_sparse_add_test 里默认带上，
    # 这里的 target 是手写的，得自己链。
    target_link_libraries(spmm_test PRIVATE
        ${OPS_SPARSE}
        $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
        ${_ops_sparse_libc_sec}
    )
else()
    message(STATUS "[test/spmm] SOC=${SOC_VERSION} not supported, skipping spmm_test")
    file(APPEND ${CMAKE_BINARY_DIR}/test/skipped_tests.list
        "spmm|SOC not supported (${SOC_VERSION})\n")
    return()
endif()

# Keep the in-tree public API ahead of any older header installed with CANN.
target_compile_options(spmm_test PRIVATE
    "SHELL:-iquote ${CMAKE_SOURCE_DIR}/include"
    "SHELL:-iquote ${CMAKE_SOURCE_DIR}/sparse/common"
    "SHELL:-iquote ${CMAKE_SOURCE_DIR}/sparse/spmm/arch22"
)
