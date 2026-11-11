# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseSpMatGetAttribute / aclsparseSpMatSetAttribute 测试：纯 Host 侧描述符
# 属性访问器，实现在 sparse/common（始终进库），无独立 kernel 源目录，故走
# NO_SRC_DIR（源码 test/spmat_get_attribute/arch35/，cmake/test.cmake 的
# NO_SRC_DIR 回退查找命中 arch35 源，先例 test/spmma_compress）。
# 本目录仅依赖 common 库符号（Create*/Get/Set/Destroy），无需内嵌其他算子源。
ops_sparse_add_gtest_tests(spmat_get_attribute ${OPS_SPARSE} NO_SRC_DIR)
# 描述符 API 测试私有共享头（test/spmat_descr_common：spmat_test_helpers /
# spmat_csv_common；目录无 CMakeLists.txt，不会被构建系统发现为测试目标）
if(TARGET spmat_get_attribute_test)
    target_include_directories(spmat_get_attribute_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../spmat_descr_common)
endif()
