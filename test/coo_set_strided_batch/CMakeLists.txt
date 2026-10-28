# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# aclsparseCooSetStridedBatch GTest tests (arch35 / ascend950)
# 注意：CSV 用例经 aclsparseSpMatGetStridedBatch 读回验证。为支持单算子独立构建测试
# （--ops=coo_set_strided_batch 不必成对指定 spmat_get_strided_batch），将配套算子的
# 实现直接编入本测试二进制：库中已有该符号时取可执行文件内副本（同源代码，行为一致）。
ops_sparse_add_gtest_tests(coo_set_strided_batch ${OPS_SPARSE} WARN_ON_MISSING_SRC)
if(TARGET coo_set_strided_batch_test)
    target_sources(coo_set_strided_batch_test PRIVATE
        ${CMAKE_SOURCE_DIR}/sparse/spmat_get_strided_batch/arch35/spmat_get_strided_batch_host.cpp)
    # 内嵌源码依赖 log/log.h（库目标有此 include，测试目标需补充）
    target_include_directories(coo_set_strided_batch_test PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common)
endif()
