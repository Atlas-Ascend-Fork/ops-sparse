# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseCreateSlicedEll / aclsparseCreateConstSlicedEll 测试：纯 Host 侧描述符
# 创建接口，实现在 sparse/common（始终进库），无独立 kernel 源目录，故走
# NO_SRC_DIR（源码 test/create_sliced_ell/arch35/）。
# TEST_P 经 aclsparseSpMatGetStridedBatch 断言默认 batchCount==1：该 getter 实现位于
# sparse/spmat_get_strided_batch/arch35/，--ops=create_sliced_ell 过滤构建时不在
# libops_sparse 中，故将其源码直接编入测试目标（同 test/csr_set_strided_batch 先例）。
ops_sparse_add_gtest_tests(create_sliced_ell ${OPS_SPARSE} NO_SRC_DIR)
if(TARGET create_sliced_ell_test)
    # 描述符 API 测试私有共享头（test/spmat_descr_common：spmat_csv_common；
    # 目录无 CMakeLists.txt，不会被构建系统发现为测试目标）
    target_include_directories(create_sliced_ell_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../spmat_descr_common)
    target_sources(create_sliced_ell_test PRIVATE
        ${CMAKE_SOURCE_DIR}/sparse/spmat_get_strided_batch/arch35/spmat_get_strided_batch_host.cpp)
    # 内嵌源码依赖 log/log.h（库目标有此 include，测试目标需补充）
    target_include_directories(create_sliced_ell_test PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common)
endif()
