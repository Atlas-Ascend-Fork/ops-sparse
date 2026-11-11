# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseDestroySpMat 测试：纯 Host 侧描述符销毁接口，实现在 sparse/common
#（始终进库），无独立 kernel 源目录，故走 NO_SRC_DIR（源码 test/destroy_spmat/arch35/）。
# 本目录仅依赖 common 库符号（Create* / Destroy），无需内嵌其他算子源。
ops_sparse_add_gtest_tests(destroy_spmat ${OPS_SPARSE} NO_SRC_DIR)
# 描述符 API 测试私有共享头（test/spmat_descr_common：spmat_test_helpers /
# spmat_csv_common；目录无 CMakeLists.txt，不会被构建系统发现为测试目标）
if(TARGET destroy_spmat_test)
    target_include_directories(destroy_spmat_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../spmat_descr_common)
endif()
