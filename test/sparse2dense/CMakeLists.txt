# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms of conditions of the CANN Open Software License Agreement Version 2.0
# ("the License");
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# SparseToDense GTest tests：框架按 SOC 架构自动选源（ascend910b / ascend910_93 → arch22，
# ascend950 → arch35）。arch22 用例为公开 API 双阶段（bit-wise golden + L2 校验链负例）；
# arch35 为既有用例，不受影响。
ops_sparse_add_gtest_tests(sparse2dense ${OPS_SPARSE}
    EXTRA_INCLUDES
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch35
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch22/op_host
)

if(TARGET sparse2dense_test)
    add_custom_command(TARGET sparse2dense_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/arch35/sparse2dense_l2_cases.csv
            $<TARGET_FILE_DIR:sparse2dense_test>/sparse2dense_l2_cases.csv
        COMMENT "Copying SparseToDense L2 CSV to build dir"
    )
endif()

# ---- arch22 Host UT（x86 可跑，无 NPU）：主线 op_host 源 + aclrt 桩 ----
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/arch22/ut/test_sparse2dense_host_ut.cpp)
    add_executable(sparse2dense_ut
        arch22/ut/test_sparse2dense_host_ut.cpp
        arch22/ut/mock/aclrt_mock.cpp
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch22/op_host/sparse2dense_tiling.cpp
    )
    target_compile_features(sparse2dense_ut PRIVATE cxx_std_17)
    target_include_directories(sparse2dense_ut PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/ut/mock
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch22/op_host
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch22/op_kernel/arch22
        ${ASCEND_CANN_PACKAGE_PATH}/include
        $ENV{LINUX_INCLUDE_PATH}
    )
    # 不链接 libascendcl：aclrt* 符号由 mock/aclrt_mock.cpp 以 Host 内存语义闭合
    target_link_libraries(sparse2dense_ut PRIVATE pthread)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        target_link_libraries(sparse2dense_ut PRIVATE GTest::gtest)
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(S2D_UT_GTEST IMPORTED_TARGET gtest)
        endif()
        if(TARGET PkgConfig::S2D_UT_GTEST)
            target_link_libraries(sparse2dense_ut PRIVATE PkgConfig::S2D_UT_GTEST)
        else()
            message(STATUS "[test/sparse2dense] GTest not found, skipping sparse2dense_ut")
            set_target_properties(sparse2dense_ut PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()
    endif()
endif()
